import numpy as np

from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.spaces.box import Box

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc.overrides import overrides
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class GaussianMLPPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            policy_parameters
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """

        name = policy_parameters["name"]
        env_spec = policy_parameters["env_spec"]
        hidden_sizes = policy_parameters["hidden_sizes"] if "hidden_sizes" in policy_parameters else (32, 32)
        learn_std = policy_parameters["learn_std"] if "learn_std" in policy_parameters else True
        init_std = policy_parameters["init_std"] if "init_std" in policy_parameters else 1.0
        adaptive_std = policy_parameters["adaptive_std"] if "adaptive_std" in policy_parameters else False
        std_share_network = policy_parameters["std_share_network"] if "std_share_network" in policy_parameters else False
        std_hidden_sizes = policy_parameters["std_hidden_sizes"] if "std_hidden_sizes" in policy_parameters else (32, 32)
        min_std = policy_parameters["min_std"] if "min_std" in policy_parameters else 1e-6
        std_hidden_nonlinearity = policy_parameters["std_hidden_nonlinearity"] if "std_hidden_nonlinearity" in policy_parameters else tf.nn.tanh
        hidden_nonlinearity = policy_parameters["hidden_nonlinearity"] if "hidden_nonlinearity" in policy_parameters else tf.nn.tanh
        output_nonlinearity = policy_parameters["output_nonlinearity"] if "output_nonlinearity" in policy_parameters else None
        mean_network = policy_parameters["mean_network"] if "mean_network" in policy_parameters else None
        std_network = policy_parameters["std_network"] if "std_network" in policy_parameters else None
        std_parametrization = policy_parameters["std_parametrization"] if "std_parametrization" in policy_parameters else 'exp'

        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        with tf.variable_scope(name):

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            # create network
            if mean_network is None:
                mean_network = MLP(
                    name="mean_network",
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                )
            self._mean_network = mean_network

            l_mean = mean_network.output_layer
            obs_var = mean_network.input_layer.input_var

            if std_network is not None:
                l_std_param = std_network.output_layer
            else:
                if adaptive_std:
                    std_network = MLP(
                        name="std_network",
                        input_shape=(obs_dim,),
                        input_layer=mean_network.input_layer,
                        output_dim=action_dim,
                        hidden_sizes=std_hidden_sizes,
                        hidden_nonlinearity=std_hidden_nonlinearity,
                        output_nonlinearity=None,
                    )
                    l_std_param = std_network.output_layer
                else:
                    if std_parametrization == 'exp':
                        init_std_param = np.log(init_std)
                    elif std_parametrization == 'softplus':
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    l_std_param = L.ParamLayer(
                        mean_network.input_layer,
                        num_units=action_dim,
                        param=tf.constant_initializer(init_std_param),
                        name="output_std_param",
                        trainable=learn_std,
                    )

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            # mean_var, log_std_var = L.get_output([l_mean, l_std_param])
            #
            # if self.min_std_param is not None:
            #     log_std_var = tf.maximum(log_std_var, np.log(min_std))
            #
            # self._mean_var, self._log_std_var = mean_var, log_std_var

            self._l_mean = l_mean
            self._l_std_param = l_std_param

            self._dist = DiagonalGaussian(action_dim)

            LayersPowered.__init__(self, [l_mean, l_std_param])
            super(GaussianMLPPolicy, self).__init__(env_spec)

            dist_info_sym = self.dist_info_sym(mean_network.input_layer.input_var, dict())
            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]

            self._f_dist = tensor_utils.compile_function(
                inputs=[obs_var],
                outputs=[mean_var, log_std_var],
            )

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None):
        mean_var, std_param_var = L.get_output([self._l_mean, self._l_std_param], obs_var)
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.std_parametrization == 'exp':
            log_std_var = std_param_var
        elif self.std_parametrization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
        else:
            raise NotImplementedError
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist
