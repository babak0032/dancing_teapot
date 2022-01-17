import argparse
import numpy as np
from multiprocessing import Queue, JoinableQueue, Process
from PIL import Image
from scipy.spatial.transform import Rotation as R
from dancing_teapot.teapot_env import render_teapot
from dancing_teapot.teapot_env_parallel import Parallel, worker_fc
from dancing_teapot.lie_tools import rodrigues, sample_matrix_threshold
import dancing_teapot.utils as utils


# def euler_angles_to_rot_matrix(theta):
#
#     # this is ZYX-intrinsic or 3-2-1 intrinsic
#     # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_%E2%86%94_Euler_angles
#     return R.from_euler('xyz', theta, degrees=False).as_matrix()

def init_episode_dict(K):
    d = {'action_matrix': []}
    for k in range(1, K+1):
        d['obs_%d' % k] = None
        d['state_matrix_%d' % k] = []
    return d

def main(args):

    parallel = Parallel(args.num_jobs, worker_fc)

    np.random.seed(args.seed)

    replay_buffer = init_episode_dict(args.K)

    # create states for workers to render
    i = 0
    limit = args.num_timesteps
    for _ in range(limit):
        # sample x0
        state = utils.sample_uniform_rotation_matrix()
        replay_buffer['state_matrix_1'].append(state)
        # sample g
        action_matrix = sample_matrix_threshold(args.min_step_size, args.max_step_size)
        replay_buffer['action_matrix'].append(action_matrix)
        # render observation
        parallel.add((state, i, args.obj_file))
        i += 1
        for k in range(2, args.K+1):
            # update state
            state = np.matmul(action_matrix, state)
            replay_buffer['state_matrix_%d' % k].append(state)
            parallel.add((state, i, args.obj_file))
            i += 1

    # render all images
    # because the next state in transition i is used as the current state in transition i + 1
    parallel.run_threads()

    # we won't be getting images in order
    for k in range(1, args.K+1):
        replay_buffer['obs_%d' % k] = [None for _ in range(limit)]

    for _ in range(args.num_timesteps * args.K):
        image, index = parallel.get()
        k = index % (args.K)
        i = int((index - k) / args.K)
        replay_buffer['obs_%d' % (k+1)][i] = image / 255.

    parallel.stop()

    # Save replay buffer to disk.
    utils.save_single_ep_h5py(replay_buffer, args.fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_timesteps', type=int, default=10,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--fname', type=str, default='data/teapot.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--obj_file', type=str, default='teapot_small.obj')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--num_jobs', type=int, default=2)
    parser.add_argument('--min-step-size', default=2 * np.pi / 4, type=float)
    parser.add_argument('--max-step-size', default=0, type=float)
    parser.add_argument('--K', required=True, type=int, help='length of sequence')
    parsed = parser.parse_args()
    main(parsed)
