import typing as T
import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

import cs285.infrastructure.utils as utils
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(
                    self.ac_dim,
                    dtype=torch.float32,
                    device=ptu.device,
                )
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with one observation to get selected action
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # Return the action that the policy prescribes
        output = self.forward(ptu.from_numpy(obs))
        if self.discrete:
            action_probs = F.log_softmax(output, dim=0).exp()
            action = torch.multinomial(action_probs, num_samples = 1)[0]
        else:
            action = torch.normal(*output)
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> T.Any:
        if self.discrete:
            return self.logits_na(observation)
        else:
            return (self.mean_net(observation), self.logstd.exp())


#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)

    def update(self, observations, actions, advantages, q_values=None, rewards=None, gamma=1):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss

        self.optimizer.zero_grad()
        out = self.forward(observations)
        if self.discrete:
            probs = F.log_softmax(out, dim=1).exp()
            log_prob = distributions.Categorical(probs=probs).log_prob(actions)
        else:
            if len(actions.shape) == 1:
                actions = actions[:, None]
            means = out[0]
            cov = torch.diag(out[1])
            log_prob = distributions.MultivariateNormal(means, cov).log_prob(actions).flatten()

        assert log_prob.shape == advantages.shape
        loss = (-log_prob * advantages).mean()
        loss.backward()
        self.optimizer.step()
        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }

        if self.nn_baseline:
            self.baseline_optimizer.zero_grad()
            ## normalize the q_values to have a mean of zero and a standard deviation of one
            ## HINT: there is a `normalize` function in `infrastructure.utils`
            q_values = utils.normalize(q_values, np.mean(q_values), np.std(q_values))
            q_values = ptu.from_numpy(q_values)
            rewards = ptu.from_numpy(rewards)

            ## use the `forward` method of `self.baseline` to get baseline predictions
            baseline_predictions = self.baseline.forward(observations).flatten()
            baseline_predictions_current = baseline_predictions[:-1]
            baseline_predictions_next = baseline_predictions[1:]
            targets = rewards[:-1] + gamma * baseline_predictions_next.detach()
            
            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            assert baseline_predictions_current.shape == targets.shape
            
            # compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            # HINT: use `F.mse_loss`
            baseline_loss = F.mse_loss(baseline_predictions_current, targets)

            # optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
            baseline_loss.backward()
            self.baseline_optimizer.step()

            train_log['Training Baseline Loss'] = ptu.to_numpy(baseline_loss)
        return train_log

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]

