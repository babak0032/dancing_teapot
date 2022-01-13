import os
import argparse
import matplotlib.pyplot as plt
import dancing_teapot.utils as utils


def main(args):
    data_path = args.fname
    replay_buffer = utils.load_list_dict_h5py(data_path)

    for i in range(len(replay_buffer['obs'])):
        print(replay_buffer['obs'][i].min(),replay_buffer['obs'][i].max())

        #print("action:",replay_buffer['action'][i])
        print("state_ids:",replay_buffer['state_matrix'][i])
        print("next_state_ids:",replay_buffer['next_state_matrix'][i])
        plt.subplot(1,2,1)
        plt.imshow(replay_buffer['obs'][i][:, :, 0], vmin=0, vmax=255, cmap="gray")
        plt.subplot(1,2,2)
        plt.imshow(replay_buffer['next_obs'][i][:, :, 0], vmin=0, vmax=255, cmap="gray")

        plt.pause(3.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('fname', type=str)
    parsed = parser.parse_args()
    main(parsed)
