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
from stable_baselines3.common.buffers import ReplayBuffer
from env import make_envs
from config import DQNConfig

conf = DQNConfig()

class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, envs.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.embedding_network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128),
            nn.LeakyReLU()
        )
        self.v_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        self.a_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, envs.single_action_space.n)
        )
    
    def forward(self, x):
        embedding = self.embedding_network(x)
        v = self.v_network(embedding)
        a = self.a_network(embedding)
        return v + (a - a.mean(dim=1, keepdim=True))

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class DQNAgent:
    def __init__(self, 
                 project_name: str = "inverted-pendulum",
                 algorithm: str = "dqn"):
        self.project_name = project_name
        self.algorithm = algorithm
    
    def create_model(self, envs):
        if self.algorithm == "dueling":
            q_network = DuelingQNetwork(envs).to(conf.device)
        else:
            q_network = QNetwork(envs).to(conf.device)
        return q_network
    
    def load_model(self, model_path: str, envs):
        if self.algorithm == "dueling":
            q_network = DuelingQNetwork(envs).to(conf.device)
        else:
            q_network = QNetwork(envs).to(conf.device)
        q_network.load_state_dict(torch.load(model_path))
        return q_network
    
    def train(self):
        # wandb init
        run = wandb.init(
            project=f"{self.project_name}-{self.algorithm}",
            config=conf.get_params_dict(),
            monitor_gym=True
        )
        
        best_reward = -float('inf')
        best_model = None
        
        # create envs
        envs = make_envs(conf.env_id, conf.n_envs)

        eval_env = make_envs(conf.env_id, 1)
        
        # create network
        q_network = self.create_model(envs)
        target_network = self.create_model(envs)
        target_network.load_state_dict(q_network.state_dict())
        
        # optimizer
        optimizer = optim.Adam(q_network.parameters(), lr=conf.learning_rate)
        
        # replay buffer
        rb = ReplayBuffer(
            conf.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            conf.device,
            n_envs=envs.num_envs,
            handle_timeout_termination=False,
        )
        
        # (n_envs, state_dim)
        obs, _ = envs.reset()
        
        for global_step in range(conf.total_timesteps):
            # calculate epsilon
            epsilon = linear_schedule(
                start_e=conf.start_epsilon,
                end_e=conf.end_epsilon,
                duration=conf.exploration_fraction * conf.total_timesteps,
                t=global_step
            )
            
            # 选择动作
            if random.random() < epsilon:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    q_values = q_network(torch.Tensor(obs).to(conf.device))
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
            if global_step > conf.learning_starts and global_step % conf.train_frequency == 0:
                data = rb.sample(conf.batch_size)
                with torch.no_grad():
                    if self.algorithm == "dqn":
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        target_values = data.rewards.flatten() + conf.gamma * target_max * (1 - data.dones.flatten())
                    else:
                        max_q_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)
                        target_values = data.rewards.flatten() \
                            + conf.gamma \
                            * target_network(data.next_observations).gather(1, max_q_actions).squeeze() \
                            * (1 - data.dones.flatten())

                # Q(s_t, a_t)
                proximate_values = q_network(data.observations).gather(1, data.actions).squeeze()
                # loss = F.mse_loss(target_values, proximate_values)
                loss = F.smooth_l1_loss(target_values, proximate_values)
                
                optimizer.zero_grad()
                loss.backward()
                # clip gradient norm
                nn.utils.clip_grad_norm_(q_network.parameters(), conf.max_grad_norm)
                optimizer.step()
                
                # 记录损失
                if global_step % 100 == 0:
                    wandb.log({
                        "losses/td_loss": loss.item(),
                        "losses/q_values": proximate_values.mean().item()
                    }, step=global_step)
            
            # 更新目标网络
            if global_step % conf.target_network_frequency == 0:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(conf.tau * param.data + (1.0 - conf.tau) * target_param.data)
            
            # eval model
            eval_reward = self.eval(global_step, eval_env, q_network)
            if eval_reward and eval_reward > best_reward:
                best_reward = eval_reward
                best_model = q_network.state_dict()
            
        # 保存模型
        dir_path = f"models/{self.project_name}/{self.algorithm}"
        os.makedirs(dir_path, exist_ok=True)
        torch.save(q_network.state_dict(), f"{dir_path}/{run.id}.pth")
        if best_model:
            torch.save(best_model, f"{dir_path}/{run.id}_best.pth")
            
        wandb.finish()
    
    def test(self, model_path: str = None):
        env = make_envs(conf.env_id, num_envs=1, render_mode="human", max_episode_steps=1000)

        q_network = self.load_model(model_path, env)
        # obs, _ = env.reset()
        obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": 0})
        iters = 0
        
        while True:
            iters += 1
            if model_path:
                with torch.no_grad():
                    q_values = q_network(torch.FloatTensor(obs).to(conf.device))
                    action = torch.argmax(q_values, dim=1).cpu().numpy()
            else:
                action = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            print(f"iter: {iters}, obs: {obs}, action: {action}, reward: {reward}")
            
            obs = next_obs
            
            env.render()[0]
            time.sleep(0.02)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            if terminated or truncated:
                break
    
    def eval(self, global_step, env, model):
        if global_step % conf.eval_frequency == 0:
            alpha_dot = np.random.uniform(-15*np.pi, 15*np.pi)
            obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": alpha_dot})
            done = False
            frames = []
            cumulative_reward = 0
            
            while not done:
                # action = render_env.action_space.sample()
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(obs).to(conf.device))
                    action = torch.argmax(q_values, dim=1).cpu().numpy()
                
                # execute action and record frame
                obs, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated
                cumulative_reward += reward
                frame = env.render()[0]
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
