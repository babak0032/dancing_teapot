from dancing_teapot.teapot_env_parallel import Parallel, worker_fc, euler_angles_to_rot_matrix
import numpy as np


def batch_renderer(seed, num_jobs, states):
    parallel = Parallel(num_jobs, worker_fc)
    np.random.seed(seed)
    parallel.add((states[0], 0))
    i = 0
    limit = states.shape[0]

    while i < limit:
        parallel.add((states[i], i + 1))
        print("iter " + str(i))
        i += 1

    parallel.run_threads()
    images = [None for _ in range(limit)]

    for _ in range(limit):
        image, index = parallel.get()
        images[index] = image

    parallel.stop()
    return images
