#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import torch
import torch.nn as nn
from gym.wrappers import TimeLimit

from salina import TAgent, instantiate_class
from salina.agents import Agents
from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch


def make_atari_env(**env_args):
    e = make_atari(env_args["env_name"])
    e = wrap_deepmind(e)
    e = wrap_pytorch(e)
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def make_gym_env(**env_args):
    e = gym.make(env_args["env_name"])
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


class PPOMLPActionAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.n
        hs = kwargs["hidden_size"]
        self.model = nn.Sequential(
            nn.Linear(input_size, hs), nn.ReLU(), nn.Linear(hs, num_outputs)
        )

    def forward(self, t, replay, stochastic, **kwargs):
        input = self.get(("env/env_obs", t))
        scores = self.model(input)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs", t), probs)

        if not replay:
            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = probs.argmax(1)
            self.set(("action", t), action)


class PPOMLPCriticAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.n
        hs = kwargs["hidden_size"]
        self.critic_model = nn.Sequential(
            nn.Linear(input_size, hs), nn.ReLU(), nn.Linear(hs, 1)
        )

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t))

        critic = self.critic_model(input).squeeze(-1)
        self.set(("critic", t), critic)


class PPOAtariActionAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = (1,) + env.observation_space.shape
        num_outputs = env.action_space.n
        self.input_shape = input_size
        #hs = kwargs["hidden_size"]
        #self.model = nn.Sequential(
        #    nn.Linear(input_size, hs), nn.ReLU(), nn.Linear(hs, num_outputs)
        #)
        self.features = nn.Sequential(
            nn.Conv2d(input_size[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.model = nn.Sequential(
            nn.Linear(self.feature_size(), 1028), nn.ReLU(), nn.Linear(1028, num_outputs)
        )

    def forward(self, t, replay, stochastic, **kwargs):
        input = self.get(("env/env_obs", t)).float()
        x = self.features(input)
        x = x.view(x.size(0), -1)
        scores = self.model(x)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs", t), probs)

        if not replay:
            if stochastic:
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = probs.argmax(1)
            self.set(("action", t), action)

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape[1:])).view(1, -1).size(1)


class PPOAtariCriticAgent(TAgent):
    def __init__(self, **kwargs):
        super().__init__()
        env = instantiate_class(kwargs["env"])
        input_size = (1,) + env.observation_space.shape
        num_outputs = env.action_space.n
        self.input_shape = input_size
        #hs = kwargs["hidden_size"]
        #self.model_critic = nn.Sequential(
        #    nn.Linear(input_size, hs), nn.ReLU(), nn.Linear(hs, 1)
        #)
        self.features = nn.Sequential(
            nn.Conv2d(input_size[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1),
        )
        self.critic_model = nn.Sequential(
            nn.Linear(self.feature_size(), 2048), nn.ReLU(), nn.Linear(2048, 1)
        )

    def forward(self, t, **kwargs):
        input = self.get(("env/env_obs", t)).float()
        x = self.features(input)
        x = x.view(x.size(0), -1)

        critic = self.critic_model(x).squeeze(-1)
        self.set(("critic", t), critic)

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape[1:])).view(1, -1).size(1)
