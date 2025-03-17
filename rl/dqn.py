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
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.buffers import ReplayBuffer
from env import InvertedPendulumEnv

class QNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.network(x)

def make_env(render_mode='rgb_array'):
    def thunk():
        env = InvertedPendulumEnv(
            render_mode=render_mode
        )
        env = RecordEpisodeStatistics(env)
        return env
    return thunk

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def train_pendulum(project_name: str = "inverted-pendulum",
                   algorithm: str = "dqn"):
    # 初始化wandb
    run = wandb.init(
        project=f"{project_name}-{algorithm}",
        config={
            "n_envs": 4,
            "total_timesteps": 500000,
            "learning_rate": 1e-4,
            "buffer_size": 2000000,
            "batch_size": 64,
            "gamma": 0.98,
            "target_network_frequency": 10000,
            "tau": 1,     # the target network update rate
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
    
    # 创建环境
    envs = SyncVectorEnv([make_env() for _ in range(config["n_envs"])])

    eval_env = InvertedPendulumEnv(
        discrete_action=True, 
        normalize_state=True, 
        render_mode='rgb_array')
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建网络
    q_network = QNetwork().to(device)
    target_network = QNetwork().to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # 优化器
    optimizer = optim.Adam(q_network.parameters(), lr=config["learning_rate"])
    
    # 经验回放缓冲区
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
        # calculate epsilon
        epsilon = linear_schedule(
            start_e=config["start_epsilon"],
            end_e=config["end_epsilon"],
            duration=config["exploration_fraction"] * config["total_timesteps"],
            t=global_step
        )
        
        # 选择动作
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        # 执行动作
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
                "charts/epsilon": epsilon
            }, step=global_step)
                
        # 记录回放缓冲区
        real_next_obs = next_obs.copy()

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        
        # update obs
        obs = next_obs
        
        # 训练
        if global_step > config["learning_starts"] and global_step % config["train_frequency"] == 0:
            data = rb.sample(config["batch_size"])
            with torch.no_grad():
                if algorithm == "dqn":
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    target_values = data.rewards.flatten() + config["gamma"] * target_max * (1 - data.dones.flatten())
                elif algorithm == "ddqn":
                    max_q_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)
                    target_values = data.rewards.flatten() \
                        + config["gamma"] \
                        * target_network(data.next_observations).gather(1, max_q_actions).squeeze() \
                        * (1 - data.dones.flatten())

            # Q(s_t, a_t)
            proximate_values = q_network(data.observations).gather(1, data.actions).squeeze()
            # loss = F.mse_loss(target_values, proximate_values)
            loss = F.smooth_l1_loss(target_values, proximate_values)
            
            optimizer.zero_grad()
            loss.backward()
            # clip gradient norm
            nn.utils.clip_grad_norm_(q_network.parameters(), config["max_grad_norm"])
            optimizer.step()
            
            # 记录损失
            if global_step % 100 == 0:
                wandb.log({
                    "losses/td_loss": loss.item(),
                    "losses/q_values": proximate_values.mean().item()
                }, step=global_step)
        
        # 更新目标网络
        if global_step % config["target_network_frequency"] == 0:
            for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(config["tau"] * param.data + (1.0 - config["tau"]) * target_param.data)
        
        # eval model
        eval_reward = eval_model(global_step, eval_env, q_network, device)
        if eval_reward and eval_reward > best_reward:
            best_reward = eval_reward
            best_model = q_network.state_dict()
        
    # 保存模型
    dir_path = f"models/{project_name}/{algorithm}"
    os.makedirs(dir_path, exist_ok=True)
    torch.save(q_network.state_dict(), f"{dir_path}/{run.id}.pth")
    if best_model:
        torch.save(best_model, f"{dir_path}/{run.id}_best.pth")
        
    wandb.finish()
    
def test_pendulum(model_path: str = None):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = InvertedPendulumEnv(max_episode_steps=500, normalize_state=True, discrete_action=True, render_mode="human")
    env = SyncVectorEnv([lambda: env])

    if model_path:
        q_network = QNetwork().to(device)
        q_network.load_state_dict(torch.load(model_path))
    obs, _ = env.reset()
    # obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": 0})
    iters = 0
    while True:
        iters += 1
        if model_path:
            with torch.no_grad():
                q_values = q_network(torch.FloatTensor(obs).to(device))
                action = torch.argmax(q_values, dim=1).cpu().numpy()
        else:
            action = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        print(f"iter: {iters}, obs: {obs}, action: {action}, reward: {reward}")
        print(f"alpha: {env.envs[0].state[0]}, alpha_dot: {env.envs[0].state[1]}")
        # 渲染当前状态
        env.render()
        
        # 添加小延时使动画更容易观察
        time.sleep(0.02)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        obs = next_obs
        if terminated or truncated:
            break
        
def eval_model(global_step, env, model, device, eval_freq=10000):
    if global_step % eval_freq == 0:
        alpha_dot = np.random.uniform(-15*np.pi, 15*np.pi)
        obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": alpha_dot})
        done = False
        frames = []
        cumulative_reward = 0
        
        while not done:
            # action = render_env.action_space.sample()
            with torch.no_grad():
                q_values = model(torch.FloatTensor(obs).to(device))
                action = torch.argmax(q_values).cpu().numpy()
            
            # execute action and record frame
            obs, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated
            cumulative_reward += reward
            frame = env.render()
            # (H, W, C) -> (C, H, W)
            frames.append(frame.transpose(2, 0, 1))
        
        video_frames = np.array(frames)
        wandb.log({
            "eval/episodic_return": cumulative_reward,
            "eval/pendulum_video": wandb.Video(
                video_frames, 
                fps=30, 
                format="gif"
            ),
        }, step=global_step)
        
        return float(cumulative_reward)
    else:
        return None
