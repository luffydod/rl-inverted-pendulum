import torch.nn as nn
import numpy as np
import random
import torch
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer

from env import InvertedPendulumEnv

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def train_pendulum():
    # 训练参数
    total_timesteps = 10000
    learning_rate = 1e-3
    buffer_size = 10000
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    target_network_frequency = 500
    learning_starts = 5000
    train_frequency = 1
    
    # 探索参数
    start_epsilon = 1.0
    end_epsilon = 0.1
    exploration_fraction = 0.5
    
    # 设置随机种子
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建环境
    env = InvertedPendulumEnv(is_discrete=True)
    n_actions = env.n_actions
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建网络
    q_network = QNetwork(env).to(device)
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # 优化器
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    
    # 经验回放缓冲区
    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    
    # Tensorboard
    run_name = f"DQN_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    # 开始训练
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    for global_step in range(total_timesteps):
        # 计算探索率
        epsilon = start_epsilon - (start_epsilon - end_epsilon) * (global_step / (exploration_fraction * total_timesteps))
        epsilon = max(epsilon, end_epsilon)
        
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
            # 记录episode信息
            writer.add_scalar("charts/episodic_return", episode_reward, episode_count)
            writer.add_scalar("charts/episodic_length", episode_length, episode_count)
            writer.add_scalar("charts/epsilon", epsilon, episode_count)
            
            print(f"Episode {episode_count}, Return: {episode_reward:.2f}, Length: {episode_length}")
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1
        else:
            obs = next_obs
        
        # 训练
        if global_step > learning_starts and global_step % train_frequency == 0:
            data = rb.sample(batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
            
            old_val = q_network(data.observations).gather(1, data.actions.long()).squeeze()
            loss = nn.MSELoss()(td_target, old_val)
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            if global_step % 1000 == 0:
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        
        # 更新目标网络
        if global_step % target_network_frequency == 0:
            for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    # 保存模型
    torch.save(q_network.state_dict(), f"runs/{run_name}/q_network.pth")
    writer.close()
    

def test_pendulum(model_path: str = None):
    if model_path is None:
        raise ValueError("model_path is None")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = InvertedPendulumEnv(is_discrete=True)
    q_network = QNetwork(env).to(device)
    q_network.load_state_dict(torch.load(model_path))
    obs, _ = env.reset()
    iters = 0
    while True:
        iters += 1
        print(f"iter: {iters}")
        print(f"obs: {obs}")
        with torch.no_grad():
            q_values = q_network(torch.FloatTensor(obs).to(device))
            action = torch.argmax(q_values).cpu().numpy()
        print(f"action: {action}")
        next_obs, reward, done, _ = env.step(action)
        print(f"reward: {reward}")
        obs = next_obs
        if done:
            break

