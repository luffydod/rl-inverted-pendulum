import time
import sys
import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import pygame
import wandb
from stable_baselines3.common.buffers import ReplayBuffer

from env import InvertedPendulumEnv

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        # 获取输入维度并转换为Python int类型
        self.input_dim = int(np.array(env.observation_space.shape).prod())
        
        # 1D卷积层部分
        self.conv_layers = nn.Sequential(
            # 确保使用Python int类型
            nn.Unflatten(1, (1, self.input_dim)),
            # 第一个卷积层
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1),
            nn.ReLU(),
            # 第二个卷积层
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1),
            nn.ReLU(),
            # 展平层
            nn.Flatten()
        )
        
        # MLP部分
        self.mlp_layers = nn.Sequential(
            nn.Linear(32 * self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n)
        )
        # self.network = nn.Sequential(
        #     nn.Linear(np.array(env.observation_space.shape).prod(), 128),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, env.action_space.n),
        # )

    def forward(self, x):
        # 确保输入形状正确
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # 添加batch维度
            
        # 通过卷积层
        x = self.conv_layers(x)
        
        # 通过MLP层
        x = self.mlp_layers(x)
        
        return x
        # return self.network(x)

def train_pendulum():
    # 初始化wandb
    run = wandb.init(
        project="inverted-pendulum",
        config={
            "total_timesteps": 500000,
            "learning_rate": 2e-4,
            "buffer_size": 10000,
            "batch_size": 128,
            "gamma": 0.99,
            "tau": 1.0,     # the target network update rate
            "target_network_frequency": 500,    # the timesteps it takes to update the target network
            "learning_starts": 10000,
            "train_frequency": 10,
            "start_epsilon": 1.0,
            "end_epsilon": 0.05,
            "exploration_fraction": 0.5, # the fraction of `total-timesteps` it takes from start epsilon to go end epsilon
        },
        monitor_gym=True
    )
    config = run.config
    
    # 创建环境
    env = InvertedPendulumEnv(discrete_action=True, render_mode='rgb_array')
    render_env = InvertedPendulumEnv(discrete_action=True, render_mode='rgb_array')
    
    n_actions = env.n_actions
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建网络
    q_network = QNetwork(env).to(device)
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # 优化器
    optimizer = optim.Adam(q_network.parameters(), lr=config["learning_rate"])
    
    # 经验回放缓冲区
    rb = ReplayBuffer(
        config["buffer_size"],
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    
    # 开始训练
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    for global_step in range(config["total_timesteps"]):
        # calculate epsilon
        epsilon = config["start_epsilon"] - (config["start_epsilon"] - config["end_epsilon"]) * (global_step / (config["exploration_fraction"] * config["total_timesteps"]))
        epsilon = max(epsilon, config["end_epsilon"])
        
        # Record video to wandb
        if global_step % 50000 == 0:
            frames = []
            obs, _ = render_env.reset()
            done = False
            
            while not done:
                # action = render_env.action_space.sample()
                with torch.no_grad():
                    q_values = q_network(torch.FloatTensor(obs).to(device))
                    action = torch.argmax(q_values).cpu().numpy()
                
                # execute action and record frame
                obs, reward, done, _ = render_env.step(action)
                frame = render_env.render()
                # (H, W, C) -> (C, H, W)
                frames.append(frame.transpose(2, 0, 1))
            
            video_frames = np.array(frames)
            # print(f"Video array shape: {video_frames.shape}, dtype: {video_frames.dtype}")
            wandb.log({
                "pendulum_video": wandb.Video(
                    video_frames, 
                    fps=30, 
                    format="mp4"
                )
            }, step=global_step)
        
        # 选择动作
        if random.random() < epsilon:
            action = np.array(random.randint(0, n_actions-1))
        else:
            with torch.no_grad():
                q_values = q_network(torch.FloatTensor(obs).to(device))
                action = torch.argmax(q_values).cpu().numpy()
        
        # 执行动作
        next_obs, reward, done, _ = env.step(action)
        
        # 记录回放缓冲区
        rb.add(obs, next_obs, action, reward, done, {})
        
        episode_reward += reward
        episode_length += 1
        
        if done:
            # record episode information to wandb
            wandb.log({
                "charts/episodic_reward": episode_reward,
                "charts/episodic_length": episode_length,
                "charts/epsilon": epsilon
            }, step=global_step)
            
            print(f"Episode {episode_count}, Reward: {episode_reward:.2f}, Length: {episode_length}")
            
            # reset environment
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1
        else:
            obs = next_obs
        
        # 训练
        if global_step > config["learning_starts"] and global_step % config["train_frequency"] == 0:
            data = rb.sample(config["batch_size"])
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + config["gamma"] * target_max * (1 - data.dones.flatten())
            
            old_val = q_network(data.observations).gather(1, data.actions.long()).squeeze()
            loss = nn.MSELoss()(td_target, old_val)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            if global_step % 1000 == 0:
                wandb.log({
                    "losses/td_loss": loss.item(),
                    "losses/q_values": old_val.mean().item()
                }, step=global_step)
        
        # 更新目标网络
        if global_step % config["target_network_frequency"] == 0:
            for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(config["tau"] * param.data + (1.0 - config["tau"]) * target_param.data)
    
    # 保存模型
    os.makedirs("models/dqn", exist_ok=True)
    torch.save(q_network.state_dict(), f"models/dqn/{run.id}.pth")
    wandb.save(f"models/dqn/{run.id}.pth")
    wandb.finish()
    

def test_pendulum(model_path: str = None):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = InvertedPendulumEnv(max_episode_steps=1000, discrete_action=True, render_mode="human")
    if model_path:
        q_network = QNetwork(env).to(device)
        q_network.load_state_dict(torch.load(model_path))
    obs, _ = env.reset()
    # obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": 0})
    iters = 0
    while True:
        iters += 1
        if model_path:
            with torch.no_grad():
                q_values = q_network(torch.FloatTensor(obs).to(device))
                action = torch.argmax(q_values).cpu().numpy()
        else:
            action = env.action_space.sample()
            
        next_obs, reward, done, _ = env.step(action)
        print(f"iter: {iters}, obs: {obs}, action: {action}, reward: {reward}")
        # 渲染当前状态
        env.render()
        
        # 添加小延时使动画更容易观察
        time.sleep(0.12)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        obs = next_obs
        if done:
            break

