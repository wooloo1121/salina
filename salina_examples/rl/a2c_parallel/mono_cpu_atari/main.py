#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import time

import gym
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TimeLimit
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
from salina import TAgent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger


from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch

def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


class A2CAgent(TAgent):
    def __init__(self, observation_size, n_actions):
        super().__init__()
        #self.model = nn.Sequential(
        #    nn.Linear(observation_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, n_actions),
        #)
        self.input_shape = observation_size
        self.features = nn.Sequential(
            nn.Conv2d(observation_size[1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.model = nn.Sequential(
            nn.Linear(self.feature_size(), 512), nn.ReLU(), nn.Linear(512, n_actions)
        )
        #self.critic_model = nn.Sequential(
        #    nn.Linear(observation_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, 1),
        #)
        self.critic_model = nn.Sequential(
            nn.Linear(self.feature_size(), 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def forward(self, t, stochastic, **kwargs):
        observation = self.get(("env/env_obs", t)).float()
        x = self.features(observation)
        x = x.view(x.size(0), -1)
        scores = self.model(x)
        probs = torch.softmax(scores, dim=-1)
        critic = self.critic_model(x).squeeze(-1)
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)
        self.set(("action_probs", t), probs)
        self.set(("critic", t), critic)

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape[1:])).view(1, -1).size(1)


def make_cartpole(max_episode_steps):
    return TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps)


def make_atari_env(max_episode_steps):
    e = make_atari("PongNoFrameskip-v4")
    e = wrap_deepmind(e)
    e = wrap_pytorch(e)
    e = TimeLimit(e, max_episode_steps=max_episode_steps)
    return e

def run_a2c(cfg):
    # 1)  Build the  logger
    logger = instantiate_class(cfg.logger)

    # 2) Create the environment agent
    # This agent implements N gym environments with auto-reset
    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=cfg.algorithm.n_envs,
    )

    # 3) Create the A2C Agent
    env = instantiate_class(cfg.algorithm.env)
    #observation_size = env.observation_space.shape[0]
    observation_size = (1,) + env.observation_space.shape
    n_actions = env.action_space.n
    del env
    a2c_agent = A2CAgent(
        observation_size, n_actions
    )

    # 4) Combine env and a2c agents
    agent = Agents(env_agent, a2c_agent)

    # 5) Get an agent that is executed on a complete workspace
    agent = TemporalAgent(agent)
    agent = agent.to(device=cfg.algorithm.device)
    agent.seed(cfg.algorithm.env_seed)

    # 6) Configure the workspace to the right dimension
    workspace = salina.Workspace()

    # 7) Confgure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer = get_class(cfg.algorithm.optimizer)(
        a2c_agent.parameters(), **optimizer_args
    )

    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        workspace.zero_grad()
        # Execute the agent on the workspace
        if epoch > 0:
            workspace.copy_n_last_steps(1)
            agent(
                workspace, t=1, n_steps=cfg.algorithm.n_timesteps - 1, stochastic=True
            )
        else:
            agent(workspace, t=0, n_steps=cfg.algorithm.n_timesteps, stochastic=True)

        # Get relevant tensors (size are timestep x n_envs x ....)
        critic, done, action_probs, reward, action = workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]

        # Compute temporal difference
        target = reward[1:] + cfg.algorithm.discount_factor * critic[1:].detach() * (
            1 - done[1:].float()
        )
        td = target - critic[:-1]

        # Compute critic loss
        td_error = td ** 2
        critic_loss = td_error.mean()

        # Compute entropy loss
        entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

        # Compute A2C loss
        action_logp = _index(action_probs, action).log()
        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = a2c_loss.mean()

        # Log losses
        logger.add_scalar("critic_loss", critic_loss.item(), epoch)
        logger.add_scalar("entropy_loss", entropy_loss.item(), epoch)
        logger.add_scalar("a2c_loss", a2c_loss.item(), epoch)

        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("reward", creward.mean().item(), epoch)


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_a2c(cfg)


if __name__ == "__main__":
    main()
