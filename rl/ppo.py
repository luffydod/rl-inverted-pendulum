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
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from datetime import datetime

from env import make_envs
from config import PPOConfig
from torch.distributions.categorical import Categorical

conf = PPOConfig()
console = Console()

# 添加初始化方法
def init_weights(module, init_type='he', gain=1.0):
    """
    初始化网络权重
    
    参数:
        module: 需要初始化的模块
        init_type: 初始化类型，可选 'he', 'xavier', 'orthogonal', 'default'
        gain: 增益因子，用于正交初始化
    """
    if isinstance(module, nn.Linear):
        if init_type == 'he':
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif init_type == 'xavier':
            nn.init.xavier_normal_(module.weight)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(module.weight, gain=gain)
        elif init_type == 'default':
            # 使用PyTorch默认初始化
            pass
        else:
            raise ValueError(f"不支持的初始化类型: {init_type}")
        
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, envs, init_type='he'):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, envs.single_action_space.n),
        )
        # 应用权重初始化
        self.apply(lambda m: init_weights(m, init_type=init_type))

    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, envs, init_type='he'):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        # 应用权重初始化
        self.apply(lambda m: init_weights(m, init_type=init_type))
        
    def forward(self, x):
        return self.network(x)

class PPOBuffer:
    """自定义PPO缓冲区实现，支持向量化环境"""
    def __init__(self, buffer_size, obs_dim, action_dim, device, n_envs=1, gae_lambda=0.95, gamma=0.99):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        
        # 计算总容量
        self.total_size = buffer_size * n_envs
        
        # 初始化缓冲区，形状为[buffer_size * n_envs, feature_dim]
        self.observations = torch.zeros((self.total_size, obs_dim), device=device)
        self.actions = torch.zeros((self.total_size,), device=device, dtype=torch.long)
        self.rewards = torch.zeros(self.total_size, device=device)
        self.values = torch.zeros(self.total_size, device=device)
        self.log_probs = torch.zeros(self.total_size, device=device)
        self.dones = torch.zeros(self.total_size, device=device, dtype=torch.bool)
        
        # 用于跟踪每个环境的指针
        self.env_ptr = 0
        self.step_ptr = 0
        self.full = False
        
    def add(self, obss, actions, rewards, dones, values, log_probs):
        """
        添加一个步骤的向量化数据到缓冲区
        
        参数:
            obss: 向量化观察 [n_envs, obs_dim]
            actions: 向量化动作 [n_envs]
            rewards: 向量化奖励 [n_envs]
            dones: 向量化终止信号 [n_envs]
            values: 向量化值函数 [n_envs, 1] 或 [n_envs]
            log_probs: 向量化对数概率 [n_envs]
        """
        # 计算当前批次在缓冲区中的索引范围
        start_idx = self.step_ptr * self.n_envs
        end_idx = start_idx + self.n_envs
        
        # 将数据添加到缓冲区
        self.observations[start_idx:end_idx] = torch.FloatTensor(obss).to(self.device)
        self.actions[start_idx:end_idx] = torch.LongTensor(actions).to(self.device)
        self.rewards[start_idx:end_idx] = torch.FloatTensor(rewards).to(self.device)
        
        # 处理值函数，确保维度正确
        if isinstance(values, torch.Tensor):
            values_tensor = values
        else:
            values_tensor = torch.FloatTensor(values).to(self.device)
            
        # 如果值函数是[n_envs, 1]形状，则压缩为[n_envs]
        if len(values_tensor.shape) > 1 and values_tensor.shape[1] == 1:
            values_tensor = values_tensor.squeeze(1)
            
        self.values[start_idx:end_idx] = values_tensor
        
        # 处理对数概率，确保维度正确
        if isinstance(log_probs, torch.Tensor):
            log_probs_tensor = log_probs
        else:
            log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
            
        # 如果对数概率是[n_envs, 1]形状，则压缩为[n_envs]
        if len(log_probs_tensor.shape) > 1 and log_probs_tensor.shape[1] == 1:
            log_probs_tensor = log_probs_tensor.squeeze(1)
            
        self.log_probs[start_idx:end_idx] = log_probs_tensor
        self.dones[start_idx:end_idx] = torch.BoolTensor(dones).to(self.device)
        
        # 更新指针
        self.step_ptr = (self.step_ptr + 1) % self.buffer_size
        self.full = self.full or self.step_ptr == 0
        
    def compute_returns_and_advantage(self, last_values, last_dones):
        """
        计算所有环境的回报和优势
        
        参数:
            last_values: 向量化最后状态的值函数 [n_envs, 1] 或 [n_envs]
            last_dones: 向量化最后状态的终止信号 [n_envs]
        """
        actual_size = self.buffer_size if self.full else self.step_ptr
        
        # 为每个环境计算回报和优势
        advantages = torch.zeros(self.total_size, device=self.device)
        returns = torch.zeros(self.total_size, device=self.device)
        
        # 确保last_values是张量并且维度正确
        if not isinstance(last_values, torch.Tensor):
            last_values = torch.FloatTensor(last_values).to(self.device)
        
        # 如果last_values是[n_envs, 1]形状，则压缩为[n_envs]
        if len(last_values.shape) > 1 and last_values.shape[1] == 1:
            last_values = last_values.squeeze(1)
        
        # 确保last_dones是张量并且维度正确
        if not isinstance(last_dones, torch.Tensor):
            last_dones = torch.BoolTensor(last_dones).to(self.device)
        
        for env_idx in range(self.n_envs):
            # 计算当前环境的数据索引
            env_indices = torch.arange(env_idx, self.total_size, self.n_envs)[:actual_size]
            if len(env_indices) == 0:
                continue
                
            # 获取当前环境的数据
            env_rewards = self.rewards[env_indices]
            env_values = self.values[env_indices]
            env_dones = self.dones[env_indices]
            
            # 准备计算GAE
            if len(last_values.shape) > 1 and last_values.shape[0] > 1:
                last_value = last_values[env_idx]
            else:
                last_value = last_values[0] if len(last_values.shape) > 0 else last_values
                
            if len(last_dones.shape) > 0 and last_dones.shape[0] > 1:
                last_done = last_dones[env_idx]
            else:
                last_done = last_dones[0] if len(last_dones.shape) > 0 else last_dones
            
            # 计算GAE
            last_gae = 0
            for step in reversed(range(len(env_indices))):
                if step == len(env_indices) - 1:
                    next_value = last_value
                    next_done = last_done
                else:
                    next_value = env_values[step + 1]
                    next_done = env_dones[step + 1]
                    
                delta = env_rewards[step] + self.gamma * next_value * (1 - float(next_done)) - env_values[step]
                last_gae = delta + self.gamma * self.gae_lambda * (1 - float(next_done)) * last_gae
                advantages[env_indices[step]] = last_gae
                
            # 计算回报
            returns[env_indices] = advantages[env_indices] + env_values
        
        # 保存到缓冲区
        self.advantages = advantages[:actual_size * self.n_envs]
        self.returns = returns[:actual_size * self.n_envs]
        
    def get_batch(self, batch_size):
        """获取小批量数据"""
        actual_size = self.buffer_size if self.full else self.step_ptr
        total_samples = actual_size * self.n_envs
        
        # 对所有样本进行随机排列
        indices = torch.randperm(total_samples, device=self.device)
        
        # 生成批次
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'observations': self.observations[batch_indices],
                'actions': self.actions[batch_indices],
                'old_values': self.values[batch_indices],
                'old_log_prob': self.log_probs[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices]
            }
            
    def reset(self):
        """重置缓冲区"""
        self.step_ptr = 0
        self.env_ptr = 0
        self.full = False

class PPOAgent:
    def __init__(self, 
                 project_name: str = None,
                 algorithm: str = "ppo",
                 init_type: str = "he"):
        self.project_name = project_name if project_name else conf.env_id
        self.algorithm = algorithm
        self.init_type = init_type
        self.actor = None
        self.critic = None
        self.console = Console()
        self.training_stats = {
            "episode_returns": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "total_losses": [],
            "learning_rates": [],
            "eval_returns": []
        }
    
    def create_model(self, envs):
        self.actor = Actor(envs, init_type=self.init_type).to(conf.device)
        self.critic = Critic(envs, init_type=self.init_type).to(conf.device)
        return self.actor, self.critic
    
    def load_model(self, model_path: str, envs):
        # 创建模型
        self.create_model(envs)
        
        # 加载模型
        model_data = torch.load(model_path, map_location=conf.device)
        self.actor.load_state_dict(model_data['actor'])
        self.critic.load_state_dict(model_data['critic'])
        
        return self.actor, self.critic
    
    def get_action_and_value(self, obs, action=None):
        """获取动作、对数概率、熵和值函数"""
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        # 获取对数概率和熵
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        # 获取值函数
        value = self.critic(obs)
        
        # 确保所有返回值的形状一致
        # 如果action是[batch_size]，则log_prob和entropy应该是[batch_size]
        # 如果value是[batch_size, 1]，则压缩为[batch_size]
        if len(value.shape) > 1 and value.shape[1] == 1:
            value = value.squeeze(1)
            
        return action, log_prob, entropy, value
    
    def _create_progress_bar(self, total_steps):
        """创建更丰富的进度条"""
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
    
    def _create_stats_table(self):
        """创建统计信息表格"""
        table = Table(title="训练统计", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right", style="green")
        return table
    
    def _update_stats_table(self, table, global_step, episode_return, episode_length, 
                           policy_loss, value_loss, entropy_loss, total_loss, 
                           learning_rate, eval_return=None):
        """更新统计信息表格"""
        table.rows = []
        table.add_row("全局步数", f"{global_step:,}")
        table.add_row("平均回报", f"{episode_return:.2f}")
        table.add_row("平均长度", f"{episode_length:.1f}")
        table.add_row("策略损失", f"{policy_loss:.4f}")
        table.add_row("值函数损失", f"{value_loss:.4f}")
        table.add_row("熵损失", f"{entropy_loss:.4f}")
        table.add_row("总损失", f"{total_loss:.4f}")
        table.add_row("学习率", f"{learning_rate:.6f}")
        if eval_return is not None:
            table.add_row("评估回报", f"{eval_return:.2f}")
        return table
    
    def _log_episode_info(self, global_step, episode_return, episode_length, episode_time):
        """记录episode信息"""
        # 更新训练统计
        self.training_stats["episode_returns"].append(episode_return)
        self.training_stats["episode_lengths"].append(episode_length)
        
        # 计算移动平均
        window_size = min(100, len(self.training_stats["episode_returns"]))
        avg_return = np.mean(self.training_stats["episode_returns"][-window_size:])
        avg_length = np.mean(self.training_stats["episode_lengths"][-window_size:])
        
        # 记录到wandb
        wandb.log({
            "charts/episodic_return": episode_return,
            "charts/episodic_length": episode_length,
            "charts/episodic_time": episode_time,
            "charts/avg_episodic_return": avg_return,
            "charts/avg_episodic_length": avg_length,
        }, step=global_step)
        
        # 打印信息
        self.console.print(f"[bold green]Episode: {global_step//episode_length} | "
                          f"Return: {episode_return:.2f} | "
                          f"Length: {episode_length:.1f} | "
                          f"Avg Return: {avg_return:.2f}[/bold green]")
    
    def _log_loss_info(self, global_step, policy_loss, value_loss, entropy_loss, total_loss, learning_rate):
        """记录损失信息"""
        # 更新训练统计
        self.training_stats["policy_losses"].append(policy_loss)
        self.training_stats["value_losses"].append(value_loss)
        self.training_stats["entropy_losses"].append(entropy_loss)
        self.training_stats["total_losses"].append(total_loss)
        self.training_stats["learning_rates"].append(learning_rate)
        
        # 计算移动平均
        window_size = min(100, len(self.training_stats["policy_losses"]))
        avg_policy_loss = np.mean(self.training_stats["policy_losses"][-window_size:])
        avg_value_loss = np.mean(self.training_stats["value_losses"][-window_size:])
        avg_entropy_loss = np.mean(self.training_stats["entropy_losses"][-window_size:])
        avg_total_loss = np.mean(self.training_stats["total_losses"][-window_size:])
        
        # 记录到wandb
        wandb.log({
            "losses/policy_loss": policy_loss,
            "losses/value_loss": value_loss,
            "losses/entropy_loss": entropy_loss,
            "losses/total_loss": total_loss,
            "losses/avg_policy_loss": avg_policy_loss,
            "losses/avg_value_loss": avg_value_loss,
            "losses/avg_entropy_loss": avg_entropy_loss,
            "losses/avg_total_loss": avg_total_loss,
            "learning_rate": learning_rate,
        }, step=global_step)
    
    def train(self, model_path: str = None):
        # 记录开始时间
        start_time = time.time()
        
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
        
        # 创建环境
        if conf.env_id == "inverted-pendulum" and model_path is not None:
            reset_option = "goal"
        else:
            reset_option = None
            
        envs = make_envs(conf.env_id, conf.n_envs, reset_option=reset_option)
        eval_env = make_envs(conf.env_id, 1)
        
        # 创建网络
        if model_path is None:
            self.create_model(envs)
        else:
            self.load_model(model_path, envs)
        
        # 优化器
        optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': conf.learning_rate},
            {'params': self.critic.parameters(), 'lr': conf.learning_rate}
        ])
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=0.1, 
            total_iters=conf.total_timesteps // conf.n_steps
        )
        
        # 创建回放缓冲区
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = envs.single_action_space.n
        rb = PPOBuffer(
            buffer_size=conf.n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=conf.device,
            n_envs=envs.num_envs,
            gae_lambda=conf.gae_lambda,
            gamma=conf.gamma
        )
        
        # 初始化观察
        obs, _ = envs.reset()
        
        # 创建进度条和统计表格
        progress = self._create_progress_bar(conf.total_timesteps)
        stats_table = self._create_stats_table()
        
        # 训练循环
        with Live(
            Panel(progress),
            refresh_per_second=4,
            console=self.console
        ) as live:
            # 创建主任务
            main_task = progress.add_task("[bold blue]训练进度", total=conf.total_timesteps)
            
            # 训练循环
            for global_step in range(0, conf.total_timesteps, conf.n_steps):
                # 更新进度条
                progress.update(main_task, completed=global_step)
                
                # 收集数据
                episode_returns = []
                episode_lengths = []
                episode_times = []
                
                for _ in range(conf.n_steps):
                    # 选择动作
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).to(conf.device)
                        action, action_log_prob, entropy, value = self.get_action_and_value(obs_tensor)
                        actions = action.cpu().numpy()
                    
                    # 执行动作
                    next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                    
                    # 记录信息
                    if "episode" in infos:
                        episode_length = infos["episode"]["l"].mean()
                        episode_return = infos["episode"]["r"].mean()
                        episode_time = infos["episode"]["t"].mean()
                        
                        episode_returns.append(episode_return)
                        episode_lengths.append(episode_length)
                        episode_times.append(episode_time)
                        
                        # 记录episode信息
                        self._log_episode_info(global_step, episode_return, episode_length, episode_time)
                    
                    # 合并terminations和truncations作为done标志
                    dones = np.logical_or(terminations, truncations)
                    
                    # 添加向量化数据到缓冲区
                    rb.add(
                        obss=obs,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        values=value,
                        log_probs=action_log_prob
                    )
                    
                    # 更新观察
                    obs = next_obs.copy()
                
                # 计算优势
                with torch.no_grad():
                    next_obs_tensor = torch.FloatTensor(obs).to(conf.device)
                    _, _, _, next_value = self.get_action_and_value(next_obs_tensor)
                    
                    # 计算所有环境的优势
                    rb.compute_returns_and_advantage(
                        last_values=next_value,
                        last_dones=dones
                    )
                
                # 训练多个epoch
                epoch_policy_losses = []
                epoch_value_losses = []
                epoch_entropy_losses = []
                epoch_total_losses = []
                
                for _ in range(conf.n_epochs):
                    # 获取小批量数据
                    for batch in rb.get_batch(conf.batch_size):
                        # 获取数据
                        obs_batch = batch['observations']
                        actions_batch = batch['actions']
                        old_values_batch = batch['old_values']
                        old_log_probs_batch = batch['old_log_prob']
                        advantages_batch = batch['advantages']
                        returns_batch = batch['returns']
                        
                        # 计算新的动作概率和值
                        _, new_log_probs, entropy, values = self.get_action_and_value(
                            obs_batch, actions_batch
                        )
                        
                        # 计算比率
                        ratio = torch.exp(new_log_probs - old_log_probs_batch)
                        
                        # 计算策略损失
                        policy_loss_1 = advantages_batch * ratio
                        policy_loss_2 = advantages_batch * torch.clamp(ratio, 1.0 - conf.clip_range, 1.0 + conf.clip_range)
                        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                        
                        # 计算值函数损失
                        if conf.clip_range_vf is None:
                            # 确保values和returns_batch的形状匹配
                            if values.shape != returns_batch.shape:
                                values = values.squeeze(1)  # 将[64, 1]压缩为[64]
                            value_loss = F.mse_loss(values, returns_batch)
                        else:
                            # 确保values和old_values_batch的形状匹配
                            if values.shape != old_values_batch.shape:
                                values = values.squeeze(1)  # 将[64, 1]压缩为[64]
                                
                            values_clipped = old_values_batch + torch.clamp(
                                values - old_values_batch, -conf.clip_range_vf, conf.clip_range_vf
                            )
                            value_loss = torch.max(
                                F.mse_loss(values, returns_batch),
                                F.mse_loss(values_clipped, returns_batch)
                            )
                        
                        # 计算熵损失
                        entropy_loss = -entropy.mean()
                        
                        # 总损失
                        loss = policy_loss + conf.vf_coef * value_loss + conf.ent_coef * entropy_loss
                        
                        # 优化
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.actor.parameters(), conf.max_grad_norm)
                        nn.utils.clip_grad_norm_(self.critic.parameters(), conf.max_grad_norm)
                        optimizer.step()
                        
                        # 记录损失
                        epoch_policy_losses.append(policy_loss.item())
                        epoch_value_losses.append(value_loss.item())
                        epoch_entropy_losses.append(entropy_loss.item())
                        epoch_total_losses.append(loss.item())
                
                # 计算平均损失
                avg_policy_loss = np.mean(epoch_policy_losses)
                avg_value_loss = np.mean(epoch_value_losses)
                avg_entropy_loss = np.mean(epoch_entropy_losses)
                avg_total_loss = np.mean(epoch_total_losses)
                
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                # 更新学习率
                scheduler.step()
                
                # 记录损失信息
                self._log_loss_info(
                    global_step, 
                    avg_policy_loss, 
                    avg_value_loss, 
                    avg_entropy_loss, 
                    avg_total_loss, 
                    current_lr
                )
                
                # 重置缓冲区
                rb.reset()
                
                # 评估模型
                eval_reward = self.eval(global_step, eval_env)
                if eval_reward is not None:
                    self.training_stats["eval_returns"].append(eval_reward)
                    
                    # 更新最佳模型
                    if eval_reward > best_reward:
                        best_reward = eval_reward
                        best_model = {
                            'actor': self.actor.state_dict(),
                            'critic': self.critic.state_dict()
                        }
                        
                        # 保存最佳模型
                        dir_path = f"models/{self.project_name}/{self.algorithm}"
                        os.makedirs(dir_path, exist_ok=True)
                        torch.save(best_model, f"{dir_path}/{run.id}_best_model.pth")
                        
                        # 打印信息
                        self.console.print(f"[bold yellow]新的最佳模型！评估回报: {eval_reward:.2f}[/bold yellow]")
                
                # 更新统计表格
                avg_episode_return = np.mean(episode_returns) if episode_returns else 0
                avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
                
                stats_table = self._update_stats_table(
                    stats_table,
                    global_step,
                    avg_episode_return,
                    avg_episode_length,
                    avg_policy_loss,
                    avg_value_loss,
                    avg_entropy_loss,
                    avg_total_loss,
                    current_lr,
                    eval_reward
                )
            
            # 完成进度条
            progress.update(main_task, completed=conf.total_timesteps)
        
        # 保存最终模型
        dir_path = f"models/{self.project_name}/{self.algorithm}"
        os.makedirs(dir_path, exist_ok=True)
        
        final_model = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'global_step': conf.total_timesteps,
            'best_reward': best_reward,
            'training_stats': self.training_stats
        }
        torch.save(final_model, f"{dir_path}/{run.id}_model.pth")
        
        # 打印训练总结
        self.console.print("\n[bold green]训练完成！[/bold green]")
        self.console.print(f"[bold green]总步数: {conf.total_timesteps:,}[/bold green]")
        self.console.print(f"[bold green]总时间: {time.time() - start_time:.1f}秒[/bold green]")
        self.console.print(f"[bold green]每秒步数: {conf.total_timesteps / (time.time() - start_time):.1f}[/bold green]")
        self.console.print(f"[bold green]最佳评估回报: {best_reward:.2f}[/bold green]")
        
        # 计算最终统计信息
        window_size = min(100, len(self.training_stats["episode_returns"]))
        final_avg_return = np.mean(self.training_stats["episode_returns"][-window_size:])
        final_avg_length = np.mean(self.training_stats["episode_lengths"][-window_size:])
        
        self.console.print(f"[bold green]最终平均回报: {final_avg_return:.2f}[/bold green]")
        self.console.print(f"[bold green]最终平均长度: {final_avg_length:.1f}[/bold green]")
        
        # 记录最终统计信息到wandb
        wandb.log({
            "final/avg_return": final_avg_return,
            "final/avg_length": final_avg_length,
            "final/best_reward": best_reward,
            "final/total_time": time.time() - start_time,
            "final/steps_per_second": conf.total_timesteps / (time.time() - start_time),
        })
        
        wandb.finish()
    
    def test(self, model_path: str = None):
        env = make_envs(conf.env_id, num_envs=1, render_mode="human")

        self.load_model(model_path, env)
        obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": 0})
        iters = 0
        total_reward = 0
        
        self.console.print("[bold blue]开始测试...[/bold blue]")
        
        while True:
            iters += 1
            if model_path:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(conf.device)
                    action, _, _, _ = self.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()
            else:
                action = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            self.console.print(f"[bold blue]步骤: {iters}, 观察: {obs}, 动作: {action}, 奖励: {reward:.2f}, 累计奖励: {total_reward:.2f}[/bold blue]")
            
            obs = next_obs
            
            env.render()[0]
            time.sleep(0.02)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            if terminated or truncated:
                break
        
        self.console.print(f"[bold green]测试完成！总步骤: {iters}, 总奖励: {total_reward:.2f}[/bold green]")
    
    def eval(self, global_step, env):
        if global_step % conf.eval_frequency != 0:
            return None
            
        self.console.print(f"[bold blue]评估模型 (步骤: {global_step})...[/bold blue]")
        
        obs, _ = env.reset(options={"alpha": np.pi, "alpha_dot": 0})
        done = False
        frames = []
        cumulative_reward = 0
        steps = 0
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(conf.device)
                action, _, _, _ = self.get_action_and_value(obs_tensor)
                action = action.cpu().numpy()
            
            # execute action and record frame
            obs, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated
            cumulative_reward += reward
            steps += 1
            
            frame = env.render()[0]
            # (H, W, C) -> (C, H, W)
            frames.append(frame.transpose(2, 0, 1))
        
        video_frames = np.array(frames)
        wandb.log({
            "eval/episodic_return": cumulative_reward,
            "eval/episodic_length": steps,
            "eval/video": wandb.Video(
                video_frames, 
                fps=30, 
                format="gif"
            ),
        }, step=global_step)
        
        # 确保cumulative_reward是浮点数
        if isinstance(cumulative_reward, np.ndarray):
            cumulative_reward = float(cumulative_reward)
            
        self.console.print(f"[bold green]评估完成！回报: {cumulative_reward:.2f}, 步骤: {steps}[/bold green]")
        
        return float(cumulative_reward)
