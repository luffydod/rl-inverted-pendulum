import time
import sys
import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pygame
import wandb
from gymnasium.vector import SyncVectorEnv
from stable_baselines3.common.buffers import ReplayBuffer
from env import make_env, make_envs

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space.shape).prod() +
                np.array(env.single_action_space.shape).prod(),
                256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, a):
        xa = torch.cat([x, a], dim=1)
        return self.network(xa)

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, np.array(env.single_action_space.shape).prod()),
            nn.Tanh(),
        )
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.network(x)
        return self.action_scale * x + self.action_bias

def train_pendulum(project_name: str = "inverted-pendulum",
                   algorithm: str = "ddpg"):
    # wandb init
    run = wandb.init(
        project=f"{project_name}-{algorithm}",
        config={
            "n_envs": 4,
            "total_timesteps": 500000,
            "learning_rate": 1e-4,
            "buffer_size": 2000000,
            "batch_size": 128,
            "gamma": 0.98,
            "target_network_frequency": 1000,
            "tau": 0.4,     # the target network update rate
            "learning_starts": 5000,
            "max_grad_norm": 10,
            "train_frequency": 4,
            "eval_frequency": 10000,
            "start_epsilon": 1.0,
            "end_epsilon": 0.05,
            "exploration_fraction": 0.2, # the fraction of `total-timesteps` it takes from start epsilon to go end epsilon
        },
        monitor_gym=True
    )
    config = run.config
    
    best_reward = -float('inf')
    best_model = None
    
    # create envs
    envs = SyncVectorEnv([make_env() for _ in range(config["n_envs"])])

    eval_env = SyncVectorEnv([make_env()])
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # create networks
    actor = Actor(envs).to(device)
    actor_target = Actor(envs).to(device)
    actor_target.load_state_dict(actor.state_dict())
    
    q_network = QNetwork(envs).to(device)
    q_target = QNetwork(envs).to(device)
    q_target.load_state_dict(q_network.state_dict())
    
    # optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=config["learning_rate"])
    q_optimizer = optim.Adam(q_network.parameters(), lr=config["learning_rate"])
    
    # replay buffer
    rb = ReplayBuffer(
        config["buffer_size"],
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=envs.num_envs,
        handle_timeout_termination=False,
    )
    
    # (n_envs, state_dim)
    obs, _ = envs.reset()
    
    for global_step in range(config["total_timesteps"]):
        # select action
        if global_step < config["learning_starts"]:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * config["exploration_noise"])
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
        
        # execute action
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        wandb.log({
            "training/avg_reward": rewards.mean(),
            "training/state_alpha": obs[:, 0].mean(),      # 角度均值
            "training/state_alpha_dot": obs[:, 1].mean(),  # 角速度均值
            "training/state_alpha_std": obs[:, 0].std(),   # 角度标准差
            "training/state_alpha_dot_std": obs[:, 1].std() # 角速度标准差
        }, step=global_step)
        
        if "episode" in infos:
            episode_length = infos["episode"]["l"].mean()
            episode_return = infos["episode"]["r"].mean() / episode_length
            episode_time = infos["episode"]["t"].mean()
            print(f"global_step={global_step}, episodic_return={episode_return}, episodic_length={episode_length}, episodic_time={episode_time}")
            # record episode information to wandb
            wandb.log({
                "charts/episodic_return": episode_return,
                "charts/episodic_length": episode_length,
                "charts/episodic_time": episode_time,
            }, step=global_step)
                
        # record replay buffer
        real_next_obs = next_obs.copy()

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        
        # update obs
        obs = next_obs
        
        # training
        if global_step > config["learning_starts"]:
            data = rb.sample(config["batch_size"])
            with torch.no_grad():
                next_state_actions = actor_target(data.next_observations)
                next_q_target = q_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + config["gamma"] * (1 - data.dones.flatten()) * (next_q_target).view(-1)
            
            q_a_values = q_network(data.observations, data.actions).view(-1)
            q_loss = F.mse_loss(next_q_value, q_a_values)
            
            q_optimizer.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), config["max_grad_norm"])
            q_optimizer.step()
            
            if global_step % config["policy_frequency"] == 0:
                actor_loss = -q_network(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), config["max_grad_norm"])
                actor_optimizer.step()
                
                # soft update target networks
                for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                    target_param.data.copy_(config["tau"] * param.data + (1.0 - config["tau"]) * target_param.data)
                for target_param, param in zip(q_target.parameters(), q_network.parameters()):
                    target_param.data.copy_(config["tau"] * param.data + (1.0 - config["tau"]) * target_param.data)
                    
            # record losses
            if global_step % 100 == 0:
                wandb.log({
                    "losses/q_loss": q_loss.item(),
                    "losses/q_values": q_a_values.mean().item(),
                    "losses/actor_loss": actor_loss.item(),
                }, step=global_step)
        
        # eval
        eval_reward = eval_model(global_step, eval_env, q_network, device)
        if eval_reward and eval_reward > best_reward:
            best_reward = eval_reward
            best_model = q_network.state_dict()
        
    # save model
    dir_path = f"models/{project_name}/{algorithm}"
    os.makedirs(dir_path, exist_ok=True)
    torch.save(q_network.state_dict(), f"{dir_path}/{run.id}.pth")
    if best_model:
        torch.save(best_model, f"{dir_path}/{run.id}_best.pth")
        
    wandb.finish()

def test(model_path: str = None):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # create envs
    envs = SyncVectorEnv([make_env() for _ in range(config["n_envs"])])
    
