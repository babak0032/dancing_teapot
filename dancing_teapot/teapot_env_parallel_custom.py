import argparse
import numpy as np
from dancing_teapot.teapot_env_parallel import Parallel, worker_fc, euler_angles_to_rot_matrix
from dancing_teapot import utils


def init_episode_dict():

    return {
        'obs': None,
        'action_matrix': [],
        'state_matrix': []
    }


def get_deltas(rad_step):

    return [(rad_step, 0, 0), (0, rad_step, 0), (0, 0, rad_step),
            (-rad_step, 0, 0), (0, -rad_step, 0), (0, 0, -rad_step)]


def main(args):

    parallel = Parallel(args.num_jobs, worker_fc)

    np.random.seed(args.seed)

    replay_buffer = []
    rad_step = 2 * np.pi / 30.0
    deltas = get_deltas(rad_step)

    # create states for workers to render
    for ep_idx in range(args.num_episodes):

        replay_buffer.append(init_episode_dict())

        state = utils.sample_uniform_rotation_matrix()
        parallel.add((state, (ep_idx, 0)))

        for step_idx in range(args.num_steps):

            # save state matrix
            replay_buffer[-1]['state_matrix'].append(state)

            if args.all_actions:
                action_matrix = utils.sample_uniform_rotation_matrix()
            else:
                # move in one of six directions by 1 / 30 of a circle
                action = np.random.randint(6)

                # turn euler angle delta into a rotation matrix
                action_euler = list(deltas[action])
                action_matrix = euler_angles_to_rot_matrix(action_euler)

            state = np.matmul(action_matrix, state)

            # save action
            replay_buffer[-1]['action_matrix'].append(action_matrix)

            # render observation
            parallel.add((state, (ep_idx, step_idx + 1)))

        replay_buffer[-1]['state_matrix'].append(state)

        if args.very_hard_hits:

            for circle_idx in range(3):

                if circle_idx == 0:
                    tmp_deltas = get_deltas(2 * np.pi / 30.)
                elif circle_idx == 1:
                    tmp_deltas = get_deltas(2 * np.pi / 8.)
                else:
                    tmp_deltas = get_deltas(2 * np.pi / 3.)

                for neg_sample_idx in range(6):

                    action_euler = list(tmp_deltas[neg_sample_idx])
                    action_matrix = euler_angles_to_rot_matrix(action_euler)
                    tmp_state = np.matmul(action_matrix, state)
                    replay_buffer[-1]['state_matrix'].append(tmp_state)
                    parallel.add((tmp_state, (ep_idx, args.num_steps + circle_idx * 6 + neg_sample_idx + 1)))

        else:

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
    tmp_num_steps = args.num_steps + 1
    if args.very_hard_hits:
        tmp_num_steps += 3 * 6
    else:
        tmp_num_steps += 6

    for ep_idx in range(args.num_episodes):

        replay_buffer[ep_idx]['obs'] = [None for _ in range(tmp_num_steps)]

    for ep_idx in range(args.num_episodes):

        print(ep_idx)

        for step_idx in range(tmp_num_steps):

            image, index = parallel.get()
            replay_buffer[index[0]]['obs'][index[1]] = image

    parallel.stop()

    # Save replay buffer to disk.
    utils.save_many_ep_h5py(replay_buffer, args.fname)


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
    parser.add_argument('--all_actions', default=False, action='store_true')
    parser.add_argument('--very_hard_hits', default=False, action='store_true')

    parsed = parser.parse_args()
    main(parsed)
