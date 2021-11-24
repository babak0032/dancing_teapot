import argparse
import numpy as np
from multiprocessing import Queue, JoinableQueue, Process
from PIL import Image
from scipy.spatial.transform import Rotation as R
from teapot_env import render_teapot
import utils


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

        rot, index = item
        image = render_teapot(rot)

        result_queue.put((image, index))


def euler_angles_to_rot_matrix(theta):

    # this is ZYX-intrinsic or 3-2-1 intrinsic
    # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_%E2%86%94_Euler_angles
    return R.from_euler('xyz', theta, degrees=False).as_matrix()


def init_episode_dict():

    return {
        'obs': None,
        'action_matrix': [],
        'next_obs': None,
        'state_matrix': [],
        'next_state_matrix': [],

        'rot_obs': None,
        'rot_action_matrix': [],
        'rot_next_obs': None,
        'rot_state_matrix': [],
        'rot_next_state_matrix': [],
        'group_action': []
    }


def main(args):

    parallel = Parallel(args.num_jobs, worker_fc)

    np.random.seed(args.seed)

    replay_buffer = init_episode_dict()
    offset = 1000000

    # [0., 0., 0.] would result in the teapot pointing down
    start_euler = [np.pi, 0., 0.]
    state = euler_angles_to_rot_matrix(start_euler)
    rad_step = 2 * np.pi / 30.0

    # state = render_teapot(alpha, beta)
    parallel.add((state, 0))

    # create states for workers to render
    i = 0
    limit = args.num_timesteps
    while i < limit:

        group_euler = [
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi / 2, np.pi / 2),
            np.random.uniform(-np.pi, np.pi)
        ]
        group_matrix = euler_angles_to_rot_matrix(group_euler)
        replay_buffer['group_action'].append(group_matrix)

        # save state matrix
        replay_buffer['state_matrix'].append(state)
        replay_buffer['rot_state_matrix'].append(np.matmul(group_matrix, state))
        parallel.add((np.matmul(group_matrix, state), i + offset))

        if args.all_actions:
            action_euler = [
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-np.pi/2, np.pi/2),
                np.random.uniform(-np.pi, np.pi)
            ]
        else:
            # move in one of six directions by 1 / 30 of a circle
            action = np.random.randint(6)
            deltas = [(rad_step, 0, 0), (0, rad_step, 0), (-rad_step, 0, 0), (0, -rad_step, 0),
                      (0, 0, rad_step), (0, 0, -rad_step)]

            # turn euler angle delta into a rotation matrix
            action_euler = list(deltas[action])

        action_matrix = euler_angles_to_rot_matrix(action_euler)

        # save action
        replay_buffer['action_matrix'].append(action_matrix)
        replay_buffer['rot_action_matrix'].append(np.matmul(group_matrix, action_matrix))

        # update state
        state = np.matmul(action_matrix, state)
        replay_buffer['next_state_matrix'].append(state)
        replay_buffer['rot_next_state_matrix'].append(np.matmul(group_matrix, state))
        parallel.add((np.matmul(group_matrix, state), i + 2 * offset))

        # render observation
        parallel.add((state, i + 1))

        #    if i % 10 == 0:
        print("iter " + str(i))
        i += 1

    # render all images
    # because the next state in transition i is used as the current state in transition i + 1
    parallel.run_threads()

    # we won't be getting images in order
    replay_buffer['obs'] = [None for _ in range(limit)]
    replay_buffer['next_obs'] = [None for _ in range(limit)]
    replay_buffer['rot_obs'] = [None for _ in range(limit)]
    replay_buffer['rot_next_obs'] = [None for _ in range(limit)]

    for _ in range(args.num_timesteps + 1 + 2 * args.num_timesteps):

        image, index = parallel.get()

        if index < offset:
            if index < args.num_timesteps:
                replay_buffer['obs'][index] = image

            if index > 0:
                replay_buffer['next_obs'][index - 1] = image
        elif index < offset * 2:
            replay_buffer['rot_obs'][index - offset] = image
        else:
            replay_buffer['rot_next_obs'][index - 2 * offset] = image

    parallel.stop()

    # Save replay buffer to disk.
    assert len(replay_buffer['obs']) == len(replay_buffer['action_matrix']) == \
        len(replay_buffer['next_obs']) == len(replay_buffer['state_matrix']) == \
        len(replay_buffer['next_state_matrix']) == \
        len(replay_buffer['rot_obs']) == len(replay_buffer['rot_action_matrix']) == \
        len(replay_buffer['rot_next_obs']) == len(replay_buffer['rot_state_matrix']) == \
        len(replay_buffer['rot_next_state_matrix'])
    utils.save_single_ep_h5py(replay_buffer, args.fname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_timesteps', type=int, default=10,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--fname', type=str, default='data/teapot.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--num_jobs', type=int, default=2)
    parser.add_argument('--all_actions', default=False, action='store_true')

    parsed = parser.parse_args()
    main(parsed)
