import argparse
import numpy as np
from multiprocessing import Queue, JoinableQueue, Process
from PIL import Image
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

        for _ in range(self.num_jobs):
            self.task_queue.put(None)

        for process in self.processes:
            process.join()


def worker_fc(task_queue, result_queue):

    while True:

        item = task_queue.get()

        if item is None:
            break

        alpha, beta, index = item
        image = render_teapot(alpha, beta)

        result_queue.put((image, index))


def main(args):

    parallel = Parallel(args.num_jobs, worker_fc)

    np.random.seed(args.seed)

    replay_buffer = {'obs': [None for _ in range(args.num_timesteps)],
                     'action': [],
                     'next_obs': [None for _ in range(args.num_timesteps)],
                     'state_ids': [],
                     'next_state_ids': []}

    alpha = np.pi / 2.0
    beta = 0
    rad_step = 2 * np.pi / 30.0

    # state = render_teapot(alpha, beta)
    parallel.add((alpha, beta, 0))

    for i in range(args.num_timesteps):
        # replay_buffer['obs'].append(state)
        replay_buffer['state_ids'].append((alpha, beta))

        action = np.random.randint(4)
        deltas = [(rad_step, 0), (0, rad_step), (-rad_step, 0), (0, -rad_step)]
        replay_buffer['action'].append(action)
        alpha += deltas[action][0]
        beta += deltas[action][1]

        # state = render_teapot(alpha, beta)
        parallel.add((alpha, beta, i + 1))
        # replay_buffer['next_obs'].append(state)
        replay_buffer['next_state_ids'].append((alpha, beta))

        #    if i % 10 == 0:
        print("iter " + str(i))

    parallel.run_threads()

    for _ in range(args.num_timesteps + 1):

        image, index = parallel.get()

        if index < args.num_timesteps:
            replay_buffer['obs'][index] = image

        if index > 0:
            replay_buffer['next_obs'][index - 1] = image

    parallel.stop()

    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_timesteps', type=int, default=10,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--fname', type=str, default='data/teapot.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--num_jobs', type=int, default=2)

    parsed = parser.parse_args()
    main(parsed)
