import atexit
from queue import Empty
from multiprocessing import Process, Queue
from rllab.sampler.utils import rollout
import numpy as np
## =======
import rllab.misc.logger as logger
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
## =======

__all__ = [
    'init_worker',
    'init_plot',
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
                ## ========
                logger.log("In stop...")
                ## ========
                break
            elif 'update' in msgs:
                ## ========
                logger.log("In update...")
                env = msgs['update'][0]
                policy = GaussianMLPPolicy(
            	    name="policy",
            	    env_spec=env.spec,
            	    # The neural network policy should have two hidden layers, each with 32 hidden units.
            	    hidden_sizes=(32, 32)
            	)
            elif 'demo' in msgs:
                ## ========
                logger.log("In demo")
                ## ========
                param_values, max_length = msgs['demo']
                policy.set_param_values(param_values)
                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
                ## ============
            elif 'pure_policy' in msgs:
                logger.log("In pure policy")
                pure_policy_data, max_length = msgs['pure_policy']
                policy.set_param_values(pure_policy_data)
                logger.log(str(pure_policy_data))
                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
                ## ============
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
    logger.log("Plot worker started")


def init_plot(env, policy):
    ## =======
    logger.log("Plot init...")
    queue.put(['update', env])
    ## =======
    # queue.put(['update', env, policy])

## ===========
def update_plot_with_pure_policy(policy_data, max_length=np.inf):
    queue.put(['pure_policy', policy_data, max_length])
## ===========
