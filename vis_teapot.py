import matplotlib.pyplot as plt
import utils

replay_buffer = utils.load_list_dict_h5py("data/teapot.h5")

for i in range(len(replay_buffer['obs'])):
    print(replay_buffer['obs'][i].min(),replay_buffer['obs'][i].max())

    print("action:",replay_buffer['action'][i])
    print("state_ids:",replay_buffer['state_ids'][i])
    print("next_state_ids:",replay_buffer['next_state_ids'][i])
    plt.subplot(1,2,1)
    plt.imshow(replay_buffer['obs'][i])
    plt.subplot(1,2,2)
    plt.imshow(replay_buffer['next_obs'][i])

    plt.show()
