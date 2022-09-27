import argparse
import numpy as np
from multiprocessing import Queue, JoinableQueue, Process
from PIL import Image
from scipy.spatial.transform import Rotation as R
from dancing_teapot.teapot_env import render_teapot
import dancing_teapot.utils as utils
from tqdm.auto import tqdm


class Parallel:

    def __init__(self, num_jobs, worker_fc):

        self.num_jobs = num_jobs
        self.worker_fc = worker_fc

        self.task_queue = Queue()
        self.result_queue = Queue()
        self.processes = []

    def run_threads(self):

        for _ in range(self.num_jobs):

            p = Process(target=self.worker_fc, args=(self.task_queue, self.result_queue))
            p.start()

            self.processes.append(p)

    def add(self, bundle):

        self.task_queue.put(bundle)

    def get(self):

        return self.result_queue.get()

    def stop(self):

        assert self.result_queue.empty()

        for _ in range(self.num_jobs):
            self.task_queue.put(None)

        for process in self.processes:
            process.join()


def worker_fc(task_queue, result_queue):

    while True:

        item = task_queue.get()

        if item is None:
            break

        rot, index, obj_file = item
        image = render_teapot(rot, obj_file=obj_file)

        result_queue.put((image, index))

def init_episode_dict(K):
    replay_buffer = {}
    for k in range(1, K+1):
        replay_buffer['obs_%d' % k] = []
        replay_buffer['action_matrix_%d' % k] = []
        replay_buffer['state_matrix_%d' % k] = []
        replay_buffer['label_%d' % k] = [] # This is basically same as state_matrix but in vector format
    del replay_buffer['action_matrix_%d' % K] # there are only K-1 steps
    return replay_buffer

def main(args):

    parallel = Parallel(args.num_jobs, worker_fc)

    np.random.seed(args.seed)

    replay_buffer = init_episode_dict(args.K)

    init_rot = np.array(eval(args.init_rot))

    # create states for workers to render
    i = 0
    limit = args.num_timesteps
    for _ in range(limit):
        # sample x0
        state, theta = utils.sample_uniform_rotation_matrix(args.group, init_rot,
                                                            args.axis_of_rotation)
        replay_buffer['state_matrix_1'].append(state)
        replay_buffer['label_1'].append(theta)
        # sample g
        action_matrix = utils.sample_matrix_threshold(args.group,args.min_step_size,
                args.max_step_size, args.axis_of_rotation)
        replay_buffer['action_matrix_1'].append(action_matrix)
        # sample acceleration if neccessary
        if args.mode == 'small_acceleration':
            acc_matrix = utils.sample_matrix_threshold(args.group,args.min_acceleration,
                    args.max_acceleration, args.axis_of_rotation)
        # render observation
        parallel.add((state, i, args.obj_file))
        i += 1
        for k in range(2, args.K+1):

            # update state
            state = np.matmul(action_matrix, state)
            replay_buffer['state_matrix_%d' % k].append(state)

            theta_k = utils.rot_matrix_to_euler_angles(state)
            if args.group == 'so2':
                theta_k = theta_k[args.axis_of_rotation]
            replay_buffer['label_%d' % k].append(theta_k)

            # update velocity if neccessary
            if k != args.K:
                if args.mode == 'small_acceleration':
                    action_matrix = np.matmul(acc_matrix, action_matrix)
                replay_buffer['action_matrix_%d' % k].append(action_matrix)

            parallel.add((state, i, args.obj_file))
            i += 1

    # render all images
    # because the next state in transition i is used as the current state in transition i + 1
    parallel.run_threads()

    # we won't be getting images in order
    for k in range(1, args.K+1):
        replay_buffer['obs_%d' % k] = [None for _ in range(limit)]

    for _ in tqdm(range(args.num_timesteps * args.K)):
        image, index = parallel.get()
        k = index % (args.K)
        i = int((index - k) / args.K)
        replay_buffer['obs_%d' % (k+1)][i] = image

    parallel.stop()

    # Save replay buffer to disk.
    utils.save_single_ep_h5py(replay_buffer, args.fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-timesteps', type=int, default=10,
                        help='Total number of sequences to simulate.')
    parser.add_argument('--fname', type=str, default='data/teapot.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--obj-file', type=str, default='teapot_small.obj')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--num-jobs', type=int, default=2)
    parser.add_argument('--min-step-size', default=0, type=float)
    parser.add_argument('--max-step-size', default=2 * np.pi / 4, type=float) # (2π / (K - 1 - 1))
    parser.add_argument('--K', required=True, type=int, help='Length of sequence')
    parser.add_argument('--group', required=True, choices=['so2', 'so3'])
    parser.add_argument('--mode', default='constant_velocity', choices=['constant_velocity', 'small_acceleration'])
    parser.add_argument('--min-acceleration', default=0, type=float)
    parser.add_argument('--max-acceleration', default=2 * np.pi / 60, type=float)
    parser.add_argument('--init-rot', default='[3.1415926, 0., 0.]', help='The base state in case the group is SO(2)')
    parser.add_argument('--axis-of-rotation', type=int, choices=[0,1,2], help='Axis of rotation in case the group is SO(2)')
    parsed = parser.parse_args()
    main(parsed)
