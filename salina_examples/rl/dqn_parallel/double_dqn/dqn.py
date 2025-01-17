#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import math
import time

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import salina
import salina.rl.functional as RLF
from salina import Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, RemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent
from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd


def run_dqn(cfg,queue,index,num_agents,seed):
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    q_agent = instantiate_class(cfg.q_agent)
    q_agent.set_name("q_agent")
    env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
    )
    pnum = len(list(q_agent.parameters()))

    q_target_agent = copy.deepcopy(q_agent)

    acq_agent = TemporalAgent(Agents(env_agent, copy.deepcopy(q_agent)))
    acq_remote_agent, acq_workspace = NRemoteAgent.create(
        acq_agent,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        epsilon=1.0,
    )
    acq_remote_agent.seed(seed)

    # == Setting up the training agents
    train_temporal_q_agent = TemporalAgent(q_agent)
    train_temporal_q_target_agent = TemporalAgent(q_target_agent)
    train_temporal_q_agent.to(cfg.algorithm.loss_device)
    train_temporal_q_target_agent.to(cfg.algorithm.loss_device)

    replay_buffer = ReplayBuffer(cfg.algorithm.buffer_size)
    acq_remote_agent(acq_workspace, t=0, n_steps=cfg.algorithm.n_timesteps, epsilon=1.0)
    replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)
    logger.message("[DDQN] Initializing replay buffer")
    while replay_buffer.size() < cfg.algorithm.initial_buffer_size:
        acq_workspace.copy_n_last_steps(cfg.algorithm.overlapping_timesteps)
        acq_remote_agent(
            acq_workspace,
            t=cfg.algorithm.overlapping_timesteps,
            n_steps=cfg.algorithm.n_timesteps - cfg.algorithm.overlapping_timesteps,
            epsilon=1.0,
        )
        replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)

    logger.message("[DDQN] Learning")
    epsilon_by_epoch = lambda epoch: cfg.algorithm.epsilon_final + (
        cfg.algorithm.epsilon_start - cfg.algorithm.epsilon_final
    ) * math.exp(-1.0 * epoch / cfg.algorithm.epsilon_exploration_decay)

    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer = get_class(cfg.algorithm.optimizer)(
        q_agent.parameters(), **optimizer_args
    )
    optimizer_t = get_class(cfg.algorithm.optimizer)(
        train_temporal_q_agent.parameters(), **optimizer_args
    )

    iteration = 0
    num_gradient = 8000 * num_agents
    count = 0
    c = 0.0
    #optimizer.zero_grad()
    gradient = [0 for i in range(pnum)]
    #for p in q_agent.parameters():
    #    #print(p.grad)
    #    gradient.append(p.grad)
    for epoch in range(cfg.algorithm.max_epoch):
        epsilon = epsilon_by_epoch(epoch)
        logger.add_scalar("monitor/epsilon", epsilon, iteration)

        for a in acq_remote_agent.get_by_name("q_agent"):
            a.load_state_dict(_state_dict(q_agent, "cpu"))

        acq_workspace.copy_n_last_steps(cfg.algorithm.overlapping_timesteps)
        acq_remote_agent(
            acq_workspace,
            t=cfg.algorithm.overlapping_timesteps,
            n_steps=cfg.algorithm.n_timesteps - cfg.algorithm.overlapping_timesteps,
            epsilon=epsilon,
        )
        replay_buffer.put(acq_workspace, time_size=cfg.algorithm.buffer_time_size)

        done, creward = acq_workspace["env/done", "env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_scalar("monitor/reward", creward.mean().item(), epoch)
            print("reward " + str(creward.mean().item()) + " " + str(epoch))

        logger.add_scalar("monitor/replay_buffer_size", replay_buffer.size(), epoch)

        # Inner loop to minimize the TD
        for inner_epoch in range(cfg.algorithm.inner_epochs):
            batch_size = cfg.algorithm.batch_size
            replay_workspace = replay_buffer.get(batch_size).to(
                cfg.algorithm.loss_device
            )
            # Batch size + Time_size
            action = replay_workspace["action"]
            train_temporal_q_agent(
                replay_workspace,
                t=0,
                n_steps=cfg.algorithm.buffer_time_size,
                replay=True,
                epsilon=0.0,
            )
            q, done, reward = replay_workspace["q", "env/done", "env/reward"]

            with torch.no_grad():
                train_temporal_q_target_agent(
                    replay_workspace,
                    t=0,
                    n_steps=cfg.algorithm.buffer_time_size,
                    replay=True,
                    epsilon=0.0,
                )
                q_target = replay_workspace["q"]

            td = RLF.doubleqlearning_temporal_difference(
                q,
                action,
                q_target,
                reward,
                done,
                cfg.algorithm.discount_factor,
            )
            error = td ** 2

            # Add burning steps for the first timesteps in the trajectories (for recurrent policies)
            burning = torch.zeros_like(td)
            burning[cfg.algorithm.burning_timesteps :] = 1.0
            error = error * burning
            loss = error.mean()
            logger.add_scalar("loss/q_loss", loss.item(), iteration)

            optimizer.zero_grad()
            loss.backward()
            #if cfg.algorithm.clip_grad > 0:
            #    n = torch.nn.utils.clip_grad_norm_(
            #        q_agent.parameters(), cfg.algorithm.clip_grad
            #    )
            #    logger.add_scalar("monitor/grad_norm", n.item(), iteration)
            #optimizer.step()

            if epoch <  3000:
                if cfg.algorithm.clip_grad > 0:
                    n = torch.nn.utils.clip_grad_norm_(
                        q_agent.parameters(), cfg.algorithm.clip_grad
                    )
                    logger.add_scalar("monitor/grad_norm", n.item(), iteration)
                optimizer.step()
            else:
                #gradient_tmp = []
                #for p in q_agent.parameters():
                #    #print(p.grad)
                #    gradient_tmp.append(p.grad)
                #gradient.append(index)
 
                #gradient = np.asarray(gradient, dtype=np.float64)
                #for i in range(num_agents):
                #print("before send")
                #queue1.send(gradient_tmp)
                if epoch % 1000 == 999:
                    i = 0
                    for p in q_agent.parameters():
                        p.grad += gradient[i]
                        i += 1
                    for p in q_agent.parameters():
                        #print(p.grad)
                        p.grad = p.grad / (c+1000.0)
                    c = 0.0
                    if cfg.algorithm.clip_grad > 0:
                        n = torch.nn.utils.clip_grad_norm_(
                            q_agent.parameters(), cfg.algorithm.clip_grad
                        )
                        logger.add_scalar("monitor/grad_norm", n.item(), iteration)
                    optimizer.step()
                if epoch % 1 == 0:
                    #optimizer.zero_grad()
                    #optimizer_t.zero_grad()
                    #c = 0.0
                    if epoch % 1000 == 0:
                        i = 0
                        for p in q_agent.parameters():
                            gradient[i] = p.grad
                            i += 1
                    else:
                        i = 0
                        for p in q_agent.parameters():
                            gradient[i] += p.grad
                            i += 1
                    while queue.poll() and c < 5000:
                        #print("before recv")
                        g = queue.recv()
                        #print(g)
                        c += 1
                        count += 1
                        #print("Get!")
                        i = 0
                        for gra in gradient:
                            gra += g[i]
                            i += 1
                gradient_tmp = []
                for p in q_agent.parameters():
                    #print(p.grad)
                    gradient_tmp.append(p.grad)
                #gradient.append(index)

                #gradient = np.asarray(gradient, dtype=np.float64)
                #for i in range(num_agents):
                #print("before send")
                queue.send(gradient_tmp)

                    #if c > 0:              
                    #    for p in q_agent.parameters():
                    #        #print(p.grad)
                    #        p.grad = p.grad / c
                    
                    #    if cfg.algorithm.clip_grad > 0:
                    #        n = torch.nn.utils.clip_grad_norm_(
                    #            q_agent.parameters(), cfg.algorithm.clip_grad
                    #        )
                    #        logger.add_scalar("monitor/grad_norm", n.item(), iteration)
                    #    optimizer.step()

#            if cfg.algorithm.clip_grad > 0:
#                n = torch.nn.utils.clip_grad_norm_(
#                    q_agent.parameters(), cfg.algorithm.clip_grad
#                )
#                logger.add_scalar("monitor/grad_norm", n.item(), iteration)
#            optimizer.step()
            iteration += 1

        # Update of the target network
        if cfg.algorithm.hard_target_update:
            if epoch % cfg.algorithm.update_target_epochs == 0:
                q_target_agent.load_state_dict(q_agent.state_dict())
        else:
            tau = cfg.algorithm.update_target_tau
            soft_update_params(q_agent, q_target_agent, tau)

    while count < num_gradient:
        g = queue[index].get()
        count += 1
        print(count)

@hydra.main(config_path=".", config_name="gym.yaml")
def main(cfg):
    import multiprocessing as mp

    mp.set_start_method("spawn")
    #logger = instantiate_class(cfg.logger)
    #logger.save_hps(cfg)
    num_agents = 2

    #q_agent = instantiate_class(cfg.q_agent)
 
    parent_conn_1, child_conn_1 = mp.Pipe()        
    parent_conn_2, child_conn_2 = mp.Pipe()
    q = []
    q.append(parent_conn_1)
    #q.append(parent_conn_2)
    #q.append(child_conn_2)
    q.append(child_conn_1)
    p = []
    #for i in range(num_agents):
    #    q.append(mp.Queue())
    for i in range(num_agents):
        p.append(mp.Process(target=run_dqn, args=(cfg,q[i],i,num_agents,432,)))
        p[i].start()
    print("start to join")
    for i in range(num_agents):
        p[i].join()
        print("joined!")

    #q_agent = instantiate_class(cfg.q_agent)
    #run_dqn(q_agent, logger, cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("plus", lambda x, y: x + y)
    OmegaConf.register_new_resolver("n_gpus", lambda x: 0 if x == "cpu" else 1)
    main()
