import argparse
import numpy as np
from multiprocessing import Queue, JoinableQueue, Process
from PIL import Image
from scipy.spatial.transform import Rotation as R
from dancing_teapot.teapot_env import render_teapot
from dancing_teapot.teapot_env_parallel import Parallel, worker_fc
from dancing_teapot.lie_tools import rodrigues, sample_matrix_threshold
import dancing_teapot.utils as utils
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch



# def euler_angles_to_rot_matrix(theta):
#
#     # this is ZYX-intrinsic or 3-2-1 intrinsic
#     # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_%E2%86%94_Euler_angles
#     return R.from_euler('xyz', theta, degrees=False).as_matrix()

def init_episode_dict(K):
    d = {'action_matrix': []}
    for k in range(1, K+1):
        d['obs_%d' % k] = []
        d['state_matrix_%d' % k] = []
        d['state_%d' % k] = []
    return d


def rotate(img, theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    zero = torch.zeros_like(theta)
    affine = torch.stack([cos, -sin, zero, sin, cos, zero], 1).view(-1, 2, 3)
    grid = F.affine_grid(affine, img.shape)
    return F.grid_sample(img, grid)[0]

def main(args):

    # parallel = Parallel(args.num_jobs, worker_fc)
    # THETA0 = np.array([-np.pi, 0, 0])
    # STATE0 = R.from_euler('xyz', THETA0, degrees=False).as_matrix()
    X0 = torch.load(args.obj_file)
    np.random.seed(args.seed)

    replay_buffer = init_episode_dict(args.K)
    print(replay_buffer)
    # create states for workers to render
    i = 0
    limit = args.num_timesteps
    for _ in tqdm(range(limit)):
        # sample x0
        theta = np.random.uniform(-np.pi, np.pi)
        state = R.from_euler('xyz', np.array([theta,0,0]), degrees=False).as_matrix()
        state_matrix = np.eye(3)
        state_matrix[0:2,0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                                          [np.sin(theta),  np.cos(theta)]])
        replay_buffer['state_matrix_1'].append(state_matrix)
        replay_buffer['state_1'].append(theta)

        obs_0 = rotate(X0[None,:], torch.FloatTensor([theta]))
        replay_buffer['obs_%d' % 1].append(np.transpose(obs_0.numpy(),(1,2,0)))

        # sample g
        action_theta = np.random.uniform(args.min_step_size, args.max_step_size)
        action_theta = np.random.choice([-1,1]) * action_theta
        action_matrix = R.from_euler('xyz', np.array([action_theta,0,0]), degrees=False).as_matrix()
        replay_buffer['action_matrix'].append(action_matrix)

        # render observation
        for k in range(2, args.K+1):
            # update state
            state = np.matmul(action_matrix, state)
            state_matrix = np.eye(3)
            state_matrix[0:2,0:2] = np.array([[np.cos(theta), -np.sin(theta)],
                                              [np.sin(theta),  np.cos(theta)]])
            replay_buffer['state_matrix_%d' % k].append(state_matrix)
            replay_buffer['state_%d' % k].append((k-1)*action_theta)
            replay_buffer['obs_%d' % k].append(np.transpose(rotate(obs_0[None,:], (k-1)*torch.FloatTensor([action_theta])).numpy(),(1,2,0)))

    # print('hey')
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
    parser.add_argument('--min-step-size', default=0, type=float)
    parser.add_argument('--max-step-size', default=2 * np.pi / 5, type=float)
    parser.add_argument('--K', required=True, type=int, help='length of sequence')
    parsed = parser.parse_args()
    main(parsed)
