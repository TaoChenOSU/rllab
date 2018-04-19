from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy


def run_task(*_):
    # Please note that different environments with different action spaces may
    # require different policies. For example with a Discrete action space, a
    # CategoricalMLPPolicy works, but for a Box action space may need to use
    # a GaussianMLPPolicy (see the trpo_gym_pendulum.py example)
    env = TfEnv(normalize(GymEnv("CartPole-v0")))

    policy_parameters = {
        "name": "policy",
        "env_spec": env.spec,
        "policy_type": CategoricalMLPPolicy,
        "hidden_sizes": (32, 32)
    }

    policy = CategoricalMLPPolicy(policy_parameters)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        policy_parameters=policy_parameters,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=2,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    plot=True,
)
