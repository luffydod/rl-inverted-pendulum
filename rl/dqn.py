import time
import sys
import os
import math
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pygame
import wandb
import rich
from buffers import ReplayBuffer, PrioritizedReplayBuffer
from env import make_envs
from config import DQNConfig
from utils import init_weights

conf = DQNConfig()
class QNetwork(nn.Module):
    def __init__(self, envs, init_type='he'):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, envs.single_action_space.n),
        )
        # 应用权重初始化
        self.apply(lambda m: init_weights(m, init_type=init_type))

    def forward(self, x):
        return self.network(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, envs, init_type='he'):
        super().__init__()
        self.embedding_network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128),
            nn.ReLU()
        )
        self.v_network = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.a_network = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, envs.single_action_space.n)
        )
        # 应用权重初始化
        self.apply(lambda m: init_weights(m, init_type=init_type))
    
    def forward(self, x):
        embedding = self.embedding_network(x)
        v = self.v_network(embedding)
        a = self.a_network(embedding)
        return v + (a - a.mean(dim=1, keepdim=True))

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def exponential_schedule(start_e: float, end_e: float, duration: int, t: int):
    lambda_ = math.log(start_e / end_e) / duration
    return max(end_e, end_e + (start_e - end_e) * math.exp(-lambda_ * t))

class DQNAgent:
    def __init__(self, 
                 project_name: str = None,
                 algorithm: str = "dqn",
                 buffer_type: str = None,
                 init_type: str = "he"):
        self.project_name = project_name if project_name else conf.env_id
        self.algorithm = algorithm
        self.buffer_type = buffer_type if buffer_type else conf.buffer_type
        self.init_type = init_type
    
    def create_model(self, envs):
        if self.algorithm == "dueling":
            q_network = DuelingQNetwork(envs, init_type=self.init_type).to(conf.device)
        else:
            q_network = QNetwork(envs, init_type=self.init_type).to(conf.device)
        return q_network
    
    def load_model(self, model_path: str, envs):
        if self.algorithm == "dueling":
            q_network = DuelingQNetwork(envs, init_type=self.init_type).to(conf.device)
        else:
            q_network = QNetwork(envs, init_type=self.init_type).to(conf.device)
        q_network.load_state_dict(torch.load(model_path, map_location=conf.device, weights_only=True))
        return q_network
    
    def train(self, model_path: str = None):
        # wandb init
        run = wandb.init(
            project=f"{self.project_name}-{self.algorithm}",
            config=conf.get_params_dict(),
            monitor_gym=True,
            settings=wandb.Settings(_disable_stats=True,
                                    _disable_meta=True),
        )
        
        best_reward = -float('inf')
        best_model = None
        
        # create envs
        if conf.env_id == "inverted-pendulum" and model_path is not None:
            reset_option = "goal"
        else:
            reset_option = None
            
        envs = make_envs(conf.env_id, conf.n_envs, normalize_obs=True, normalize_reward=True, reset_option=reset_option)
        eval_env = make_envs(conf.env_id, 1, normalize_obs=True, normalize_reward=True)
        
        # create network
        if model_path is None:
            q_network = self.create_model(envs)
        else:
            q_network = self.load_model(model_path, envs)
        target_network = self.create_model(envs)
        target_network.load_state_dict(q_network.state_dict())
        
        # optimizer
        optimizer = optim.Adam(q_network.parameters(), lr=conf.learning_rate)
        
        if self.buffer_type == "per":
            # prioritized experience replay buffer
            rb = PrioritizedReplayBuffer(
                conf.buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                conf.device,
                n_envs=envs.num_envs,
            )
        else:
            # replay buffer
            rb = ReplayBuffer(
                conf.buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                conf.device,
                n_envs=envs.num_envs,
            )
        
        # (n_envs, state_dim)
        obs, _ = envs.reset()
        
        for global_step in rich.progress.track(range(conf.total_timesteps), description="Training..."):
            # calculate epsilon
            # epsilon = linear_schedule(
            #     start_e=conf.start_epsilon,
            #     end_e=conf.end_epsilon,
            #     duration=conf.exploration_fraction * conf.total_timesteps,
            #     t=global_step
            # )
            epsilon = exponential_schedule(
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
            
            dones = np.logical_or(terminations, truncations)
            if "episode" in infos:
                episode_length = infos["episode"]["l"].mean()
                episode_return = infos["episode"]["r"].mean()
                episode_time = infos["episode"]["t"].mean()
                print(f"step={global_step}, episodic_return={episode_return:.2f}, episodic_length={episode_length:.2f}")
                # record episode information to wandb
                wandb.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_length,
                    "charts/episodic_time": episode_time,
                    "charts/epsilon": epsilon
                }, step=global_step)
                    
            # 记录回放缓冲区
            real_next_obs = next_obs.copy()

            rb.add(obs, real_next_obs, actions, rewards, dones, infos)
            
            # update obs
            obs = next_obs
            
            # 训练
            if global_step > conf.learning_starts and global_step % conf.train_frequency == 0:
                if self.buffer_type == "per":
                    # 对于PER，sample返回(samples, indices, weights)
                    data, indices, weights = rb.sample(conf.batch_size)
                else:
                    # 对于普通ReplayBuffer，sample只返回samples
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
                
                if self.buffer_type == "uniform":
                    loss = F.mse_loss(target_values, proximate_values)
                    # loss = F.smooth_l1_loss(target_values, proximate_values)
                # PER
                else:
                    with torch.no_grad():
                        td_errors = torch.abs(target_values - proximate_values)
                        # clip td_errors
                        td_errors = torch.clamp(td_errors, min=0, max=100)
                    # 使用IS权重计算加权损失
                    weights_tensor = torch.FloatTensor(weights).to(conf.device)
                    loss = (F.mse_loss(target_values, proximate_values, reduction='none') * weights_tensor).mean()
                    # 更新优先级
                    rb.update_priorities(indices, td_errors.detach().cpu().numpy())
                
                optimizer.zero_grad()
                loss.backward()
                # clip gradient norm
                nn.utils.clip_grad_norm_(q_network.parameters(), conf.max_grad_norm)
                optimizer.step()
                
                # 记录损失
                if global_step % 200 == 0:
                    log_data = {
                        "losses/td_loss": loss.item(),
                        "losses/q_values": proximate_values.mean().item()
                    }
                    if self.buffer_type == "per":
                        log_data.update({
                            "per/beta": rb.beta,
                            "per/mean_weight": weights_tensor.mean().item(),
                            "per/max_priority": rb.max_priority
                        })
                    wandb.log(log_data, step=global_step)
            
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
        if best_model is not None:
            torch.save(best_model, f"{dir_path}/{run.id}_best.pth")
            
        wandb.finish()
    
    def test(self, model_path: str = None):
        env = make_envs(conf.env_id, num_envs=1, render_mode="human")

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
        if global_step % conf.eval_frequency != 0:
            return None
        obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": 0})
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
            
            done = terminated or truncated
            cumulative_reward += reward
            frame = env.render()[0]
            # (H, W, C) -> (C, H, W)
            frames.append(frame.transpose(2, 0, 1))
        
        video_frames = np.array(frames)
        wandb.log({
            "eval/episodic_return": cumulative_reward,
            "eval/video": wandb.Video(
                video_frames, 
                fps=30, 
                format="gif"
            ),
        }, step=global_step)
        
        return float(cumulative_reward)
