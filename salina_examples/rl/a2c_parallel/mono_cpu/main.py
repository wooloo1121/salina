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


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


class A2CAgent(TAgent):
    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.critic_model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, t, stochastic, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        critic = self.critic_model(observation).squeeze(-1)
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)
        self.set(("action_probs", t), probs)
        self.set(("critic", t), critic)


def make_cartpole(max_episode_steps):
    return TimeLimit(gym.make("CartPole-v0"), max_episode_steps=max_episode_steps)


def run_a2c(cfg,q,n,num_agents):
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
    observation_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    del env
    a2c_agent = A2CAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )

    # 4) Combine env and a2c agents
    agent = Agents(env_agent, a2c_agent)

    # 5) Get an agent that is executed on a complete workspace
    agent = TemporalAgent(agent)
    agent.seed(cfg.algorithm.env_seed)

    # 6) Configure the workspace to the right dimension
    workspace = salina.Workspace()

    # 7) Confgure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer = get_class(cfg.algorithm.optimizer)(
        a2c_agent.parameters(), **optimizer_args
    )

    num_gradient = 5000 * num_agents
    count = 0
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

        if epoch < 5000:
            optimizer.step()
        else:
            gradient = []
            for p in a2c_agent.parameters():
                gradient.append(p.grad)

            for i in range(num_agents):
            	q[i].put(gradient)
            if epoch % 100 == 99:
                optimizer.zero_grad()
                c = 0.0
                while not q[n].empty() and c < 100:
                    g = q[n].get()
                    c += 1
                    count += 1
                    print("Get!")
                    i = 0
                    for p in a2c_agent.parameters():
                        p.grad += g[i]
                        i += 1
                for p in a2c_agent.parameters():
                    p.grad = p.grad / c
                optimizer.step()
#                while not q[n].empty():
#                    g = q[n].get()
#                    print("Get!")
#                    count += 1

        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("reward", creward.mean().item(), epoch)
            print("epoch end")

#    count_start = count
#    optimizer.zero_grad()
    while count < num_gradient:
        g = q[n].get()
        count += 1
        print(count)
#        i = 0
#        for p in a2c_agent.parameters():
#            p.grad += g[i]
#            i += 1
#        if (count - count_start)%100 == 0 or count == num_gradient:
#            epoch += 1
#            if (count - count_start)%100 == 0:
#                for p in a2c_agent.parameters():
#                    p.grad = p.grad / 100.0
#            else:
#                c = (count - count_start)%100
#                for p in a2c_agent.parameters():
#                    p.grad = p.grad / float(c)
#            optimizer.step()
#            optimizer.zero_grad()
#            workspace.zero_grad()
#            # Execute the agent on the workspace
#            workspace.copy_n_last_steps(1)
#            agent(
#                  workspace, t=1, n_steps=cfg.algorithm.n_timesteps - 1, stochastic=True
#            )
#            creward = workspace["env/cumulated_reward"]
#            creward = creward[done]
#            if creward.size()[0] > 0:
#                logger.add_scalar("reward", creward.mean().item(), epoch) 



@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    num_agents = 6

    q = []
    p = []
    for i in range(num_agents):
        q.append(mp.Queue())
    for i in range(num_agents):
        p.append(mp.Process(target=run_a2c, args=(cfg,q,i,num_agents,)))
        p[i].start()
    print("start to join")
    for i in range(num_agents):
        p[i].join()
        print("joined!")

    #run_a2c(cfg)


if __name__ == "__main__":
    main()
