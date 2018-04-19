from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite

def run_task(*_):
	env = TfEnv(normalize(CartpoleEnv()))

	policy_parameters = {
		"name": "policy",
	    "env_spec": env.spec,
	    "policy_type": GaussianMLPPolicy,
	    "hidden_sizes": (32, 32)
	}

	policy = GaussianMLPPolicy(policy_parameters)

	baseline = LinearFeatureBaseline(env_spec=env.spec)

	algo = TRPO(
	    env=env,
	    policy=policy,
	    policy_parameters=policy_parameters,
	    baseline=baseline,
	    batch_size=4000,
	    max_path_length=100,
	    n_itr=40,
	    discount=0.99,
	    step_size=0.01,
	    plot=True,
	    # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

	)
	algo.train()

run_experiment_lite(
	run_task,
	n_parallel=4,
	snapshot_mode="last",
	seed=1,
	plot=True,
)
