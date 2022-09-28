from dancing_teapot.teapot_sequence_env_parallel import Parallel, worker_fc
from tqdm.auto import tqdm
import numpy as np


def batch_renderer(seed, num_jobs, states, obj_file='teapot_small.obj'):
    parallel = Parallel(num_jobs, worker_fc)
    np.random.seed(seed)
    # parallel.add((states[0], 0))
    i = 0
    limit = states.shape[0]

    while i < limit:
        parallel.add((states[i], i, obj_file))
        i += 1

    parallel.run_threads()
    images = [None for _ in range(limit)]

    for _ in tqdm(range(limit)):
        image, index = parallel.get()
        images[index] = image

    parallel.stop()
    return images
