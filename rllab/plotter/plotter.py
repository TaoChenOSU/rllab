import atexit
from queue import Empty
from multiprocessing import Process, Queue
from rllab.sampler.utils import rollout
import numpy as np
import rllab.misc.logger as logger
import sandbox.rocky.tf.policies

__all__ = [
    'init_worker',
    'init_plot',
    'init_plot_tf',
    'update_plot',
    'update_plot_with_pure_policy'
]

process = None
queue = None


def _worker_start():
    env = None
    policy = None
    max_length = None
    try:
        while True:
            msgs = {}
            # Only fetch the last message of each type
            while True:
                try:
                    msg = queue.get_nowait()
                    msgs[msg[0]] = msg[1:]
                except Empty:
                    break
            if 'stop' in msgs:
                break
            elif 'update' in msgs:
                env, policy = msgs['update']
            elif 'update_tf' in msgs:
                env, policy_parameters = msgs['update_tf']
                policy = policy_parameters["policy_type"](policy_parameters)
            elif 'demo' in msgs:
                param_values, max_length = msgs['demo']
                policy.set_param_values(param_values)
                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
            elif 'pure_policy' in msgs:
                pure_policy_data, max_length = msgs['pure_policy']
                policy.set_param_values(pure_policy_data)
                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
            else:
                if max_length:
                    rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
    except KeyboardInterrupt:
        pass


def _shutdown_worker():
    if process:
        queue.put(['stop'])
        queue.close()
        process.join()


def init_worker():
    global process, queue
    queue = Queue()
    process = Process(target=_worker_start)
    process.start()
    atexit.register(_shutdown_worker)

def init_plot(env, policy):
    queue.put(['update', env, policy])

def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])

def init_plot_tf(env, policy_parameters):
    queue.put(['update_tf', env, policy_parameters])

def update_plot_with_pure_policy(policy_data, max_length=np.inf):
    queue.put(['pure_policy', policy_data, max_length])
