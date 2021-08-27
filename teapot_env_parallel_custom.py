import argparse
import numpy as np
from teapot_env_parallel import Parallel, worker_fc, euler_angles_to_rot_matrix
import utils


def init_episode_dict():

    return {
        'obs': None,
        'action': [],
        'action_matrix': [],
        'state_matrix': []
    }


def main(args):

    parallel = Parallel(args.num_jobs, worker_fc)

    np.random.seed(args.seed)

    replay_buffer = []
    rad_step = 2 * np.pi / 30.0
    deltas = [(rad_step, 0, 0), (0, rad_step, 0), (-rad_step, 0, 0), (0, -rad_step, 0),
              (0, 0, rad_step), (0, 0, -rad_step)]

    # create states for workers to render
    for ep_idx in range(args.num_episodes):

        replay_buffer.append(init_episode_dict())

        state = euler_angles_to_rot_matrix(
            [np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(-np.pi, np.pi)]
        )
        parallel.add((state, (ep_idx, 0)))

        for step_idx in range(args.num_steps):

            # save state matrix
            replay_buffer[-1]['state_matrix'].append(state)

            # move in one of six directions by 1 / 30 of a circle
            action = np.random.randint(6)

            # turn euler angle delta into a rotation matrix
            action_euler = list(deltas[action])
            action_matrix = euler_angles_to_rot_matrix(action_euler)

            # save action
            replay_buffer[-1]['action'].append(action)
            replay_buffer[-1]['action_matrix'].append(action_matrix)

            # render observation
            parallel.add((state, (ep_idx, step_idx + 1)))

        replay_buffer[-1]['state_matrix'].append(state)

        for neg_sample_idx in range(6):

            action_euler = list(deltas[neg_sample_idx])
            action_matrix = euler_angles_to_rot_matrix(action_euler)
            tmp_state = np.matmul(action_matrix, state)
            replay_buffer[-1]['state_matrix'].append(tmp_state)
            parallel.add((tmp_state, (ep_idx, args.num_steps + neg_sample_idx + 1)))

    # render all images
    # we will render even images in the blacklist
    # because the next state in transition i is used as the current state in transition i + 1
    parallel.run_threads()

    # we won't be getting images in order
    for ep_idx in range(args.num_episodes):

        replay_buffer[ep_idx]['obs'] = [None for _ in range(args.num_steps + 7)]

    for ep_idx in range(args.num_episodes):

        for step_idx in range(args.num_steps + 7):

            image, index = parallel.get()
            replay_buffer[index[0]]['obs'][index[1]] = image

    parallel.stop()

    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_steps', type=int, default=1,
                        help='Total number of episodes to simulate.')
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument('--fname', type=str, default='data/teapot_custom_valid.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--num_jobs', type=int, default=2)

    parsed = parser.parse_args()
    main(parsed)
