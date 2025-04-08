import numpy as np
import torch as th
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Type, Union, Any
from gymnasium import spaces
from buffers import RolloutBuffer
from utils import explained_variance, get_schedule_fn, obs_as_tensor
from utils import get_device, init_weights
import wandb
import os
import rich.progress
from env import make_envs
from config import PPOConfig
conf = PPOConfig()

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

# 自定义ActorCriticPolicy类，替代stable_baselines3的依赖
class ActorCriticPolicy(nn.Module):
    """
    自定义的Actor-Critic策略网络
    
    参数:
        envs: 环境
        learning_rate: 学习率
        init_type: 权重初始化类型
    """
    def __init__(self, envs, learning_rate=3e-4, init_type='he'):
        super().__init__()
        
        # 创建Actor和Critic网络
        self.actor = Actor(envs, init_type=init_type)
        self.critic = Critic(envs, init_type=init_type)
        
        # 设置优化器
        self.optimizer = th.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 保存环境信息
        self.observation_space = envs.single_observation_space
        self.action_space = envs.single_action_space
        
        # 训练模式标志
        self._training = True
        
    def forward(self, obs):
        """
        前向传播，返回动作、价值和动作概率的对数
        
        参数:
            obs: 观察值
            
        返回:
            actions: 动作
            values: 价值
            log_probs: 动作概率的对数
        """
        # 获取动作分布
        action_logits = self.actor(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 采样动作
        dist = th.distributions.Categorical(action_probs)
        actions = dist.sample()
        
        # 计算动作概率的对数
        log_probs = dist.log_prob(actions)
        
        # 计算价值
        values = self.critic(obs)
        
        return actions, values, log_probs
    
    def evaluate_actions(self, obs, actions):
        """
        评估动作，返回价值、动作概率的对数和熵
        
        参数:
            obs: 观察值
            actions: 动作
            
        返回:
            values: 价值
            log_probs: 动作概率的对数
            entropy: 熵
        """
        # 获取动作分布
        action_logits = self.actor(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 创建分布
        dist = th.distributions.Categorical(action_probs)
        
        # 计算动作概率的对数
        log_probs = dist.log_prob(actions)
        
        # 计算熵
        entropy = dist.entropy()
        
        # 计算价值
        values = self.critic(obs)
        
        return values, log_probs, entropy
    
    def predict_values(self, obs):
        return self.critic(obs)
    
    def obs_to_tensor(self, obs):
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        return obs.unsqueeze(0)
    
    def set_training_mode(self, training):
        self._training = training
        self.train(training)
    
class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 算法实现
    
    参数:
        policy: 策略模型
        env: 环境
        project_name: 项目名称
        algorithm: 算法名称
    """
    
    def __init__(
        self,
        policy: ActorCriticPolicy = None,
        env: gym.Env = None,
        project_name: str = None,
        algorithm: str = "ppo",
    ):
        self.project_name = project_name if project_name else conf.env_id
        self.algorithm = algorithm
        self.policy = policy
        self.env = env
        
        # 从配置中读取超参数
        self.learning_rate = conf.learning_rate
        self.n_steps = conf.n_steps
        self.batch_size = conf.batch_size
        self.n_epochs = conf.n_epochs
        self.gamma = conf.gamma
        self.gae_lambda = conf.gae_lambda
        self.clip_range = conf.clip_range
        self.clip_range_vf = conf.clip_range_vf
        self.ent_coef = conf.ent_coef
        self.vf_coef = conf.vf_coef
        self.max_grad_norm = conf.max_grad_norm
        self.target_kl = conf.target_kl
        self.normalize_advantage = conf.normalize_advantage
        
        # 设置设备
        self.device = get_device(conf.device)
        
    def _setup_model(self) -> None:
        """设置模型"""
        self._setup_lr_schedule()
        
    def _setup_lr_schedule(self) -> None:
        """设置学习率调度器"""
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        
    def _update_learning_rate(self, optimizer: th.optim.Optimizer) -> None:
        """更新学习率"""
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_schedule(self._current_progress_remaining)
            
    def collect_rollouts(
        self,
        n_rollout_steps: int = None,
    ) -> bool:
        """
        收集经验并填充RolloutBuffer
        
        参数:
            n_rollout_steps: 收集的步数
            
        返回:
            是否成功收集了足够的步数
        """
        if n_rollout_steps is None:
            n_rollout_steps = self.n_steps
            
        assert self._last_obs is not None, "没有提供之前的观察"
        
        # 切换到评估模式
        self.policy.set_training_mode(False)
        
        n_steps = 0
        self.rollout_buffer.reset()
            
        while n_steps < n_rollout_steps:
            with th.no_grad():
                # 转换为PyTorch张量
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            
            # 执行动作
            new_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            
            self.num_timesteps += self.env.num_envs
                    
            # 添加到缓冲区
            self.rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            
            # 记录episode信息到wandb
            if "episode" in infos:
                episode_length = infos["episode"]["l"].mean()
                episode_return = infos["episode"]["r"].mean()
                episode_time = infos["episode"]["t"].mean()
                print(f"global_step={self.num_timesteps}, episodic_return={episode_return:.2f}")
                # 记录episode信息到wandb
                wandb.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_length,
                    "charts/episodic_time": episode_time,
                }, step=self.num_timesteps)
            
            self._last_obs = new_obs
            self._last_episode_starts = terminations
            n_steps += 1
            
        with th.no_grad():
            # 计算最后一步的价值
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            
        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=terminations)
        
        return True
    
    def train_step(self) -> None:
        """
        使用当前收集的rollout数据更新策略参数
        """
        # 切换到训练模式
        self.policy.set_training_mode(True)
        
        # 更新优化器学习率
        self._update_learning_rate(self.policy.optimizer)
        
        # 计算当前裁剪范围
        clip_range = self.clip_range
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf
            
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        
        continue_training = True
        
        # 训练n_epochs轮
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # 对rollout缓冲区进行完整遍历
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.env.single_action_space, spaces.Discrete):
                    # 将离散动作从float转换为long
                    actions = rollout_data.actions.long().flatten()
                    
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                
                # 归一化优势
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                # 新旧策略的比率，第一次迭代应该为1
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                
                # 裁剪的替代损失
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                
                # 记录
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                if self.clip_range_vf is None:
                    # 不裁剪
                    values_pred = values
                else:
                    # 裁剪新旧价值的差异
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    
                # 使用TD(gae_lambda)目标的价值损失
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                
                # 熵损失，鼓励探索
                if entropy is None:
                    # 当没有解析形式时近似熵
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                    
                entropy_losses.append(entropy_loss.item())
                
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # 计算近似形式的反向KL散度，用于早停
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                    
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    print(f"在步骤{epoch}处早停，因为达到了最大kl: {approx_kl_div:.2f}")
                    break
                    
                # 优化步骤
                self.policy.optimizer.zero_grad()
                loss.backward()
                # 裁剪梯度范数
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
            self._n_updates += 1
            if not continue_training:
                break
                
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        
        # 记录日志
        print(f"训练/熵损失: {np.mean(entropy_losses):.4f}")
        print(f"训练/策略梯度损失: {np.mean(pg_losses):.4f}")
        print(f"训练/价值损失: {np.mean(value_losses):.4f}")
        print(f"训练/近似kl: {np.mean(approx_kl_divs):.4f}")
        print(f"训练/裁剪比例: {np.mean(clip_fractions):.4f}")
        print(f"训练/损失: {loss.item():.4f}")
        print(f"训练/解释方差: {explained_var:.4f}")
        print(f"训练/更新次数: {self._n_updates}")
        print(f"训练/裁剪范围: {clip_range}")
        if self.clip_range_vf is not None:
            print(f"训练/价值函数裁剪范围: {clip_range_vf}")
            
        # 记录到wandb
        wandb.log({
            "losses/entropy_loss": np.mean(entropy_losses),
            "losses/policy_loss": np.mean(pg_losses),
            "losses/value_loss": np.mean(value_losses),
            "losses/approx_kl": np.mean(approx_kl_divs),
            "losses/clip_fraction": np.mean(clip_fractions),
            "losses/loss": loss.item(),
            "losses/explained_variance": explained_var,
            "train/n_updates": self._n_updates,
            "train/clip_range": clip_range,
        }, step=self.num_timesteps)
    
    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
    ):
        """
        学习指定数量的时间步
        
        参数:
            total_timesteps: 总时间步数
            log_interval: 日志记录间隔
            reset_num_timesteps: 是否重置时间步计数
            
        返回:
            self
        """
        if reset_num_timesteps:
            self.num_timesteps = 0
            
        iteration = 0
        
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts()
            
            if not continue_training:
                break
                
            iteration += 1
            
            # 显示训练信息
            if log_interval is not None and iteration % log_interval == 0:
                print(f"迭代: {iteration}, 时间步: {self.num_timesteps}/{total_timesteps}")
                
            self.train_step()
            
        return self 
        
    def train(self, model_path: str = None):
        """
        训练PPO代理
        
        参数:
            model_path: 预训练模型路径，如果为None则创建新模型
        """
        # 初始化wandb
        run = wandb.init(
            project=f"{self.project_name}-{self.algorithm}",
            config=conf.get_params_dict(),
            monitor_gym=True,
            settings=wandb.Settings(_disable_stats=True,
                                    _disable_meta=True),
        )
        
        best_reward = -float('inf')
        best_model = None
        
        envs = make_envs(conf.env_id, conf.n_envs)
        eval_env = make_envs(conf.env_id, 1)
        
        # 创建策略网络
        if model_path is None:
            self.policy = ActorCriticPolicy(envs, learning_rate=conf.learning_rate)
        else:
            self.policy = ActorCriticPolicy(envs, learning_rate=conf.learning_rate)
            self.policy.load_state_dict(th.load(model_path, map_location=conf.device))
            
        # 设置设备
        self.policy = self.policy.to(self.device)
        
        # 初始化环境
        self.env = envs
        
        # 初始化缓冲区
        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.env.single_observation_space,
            self.env.single_action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.env.num_envs,
        )
        
        # 训练状态
        self.num_timesteps = 0
        self._last_obs = None
        self._last_episode_starts = None
        self._n_updates = 0
        self._current_progress_remaining = 1.0
        
        # 设置学习率调度器
        self._setup_lr_schedule()
        
        # 训练循环
        iteration = 0
        
        # 获取初始观察
        obs, _ = envs.reset()
        self._last_obs = obs
        self._last_episode_starts = np.zeros(envs.num_envs, dtype=bool)
        
        for global_step in rich.progress.track(range(conf.total_timesteps), description="Training..."):
            # 收集经验
            continue_training = self.collect_rollouts()
            
            if not continue_training:
                break
                
            iteration += 1
            
            # 训练策略
            self.train_step()
            
            # 显示训练信息
            if conf.log_interval is not None and iteration % conf.log_interval == 0:
                print(f"迭代: {iteration}, 时间步: {self.num_timesteps}/{conf.total_timesteps}")
                
            # 评估模型
            if iteration % conf.eval_frequency == 0:
                eval_reward = self.eval(global_step, eval_env)
                if eval_reward and eval_reward > best_reward:
                    best_reward = eval_reward
                    best_model = self.policy.state_dict()
                    
                # 记录评估结果到wandb
                wandb.log({
                    "charts/eval_reward": eval_reward,
                }, step=global_step)
        
        # 保存模型
        dir_path = f"models/{self.project_name}/{self.algorithm}"
        os.makedirs(dir_path, exist_ok=True)
        th.save(self.policy.state_dict(), f"{dir_path}/{run.id}.pth")
        if best_model is not None:
            th.save(best_model, f"{dir_path}/{run.id}_best.pth")
            
        wandb.finish()
        
    def eval(self, global_step, eval_env):
        """
        评估当前策略
        
        参数:
            global_step: 当前全局步数
            eval_env: 评估环境
            
        返回:
            eval_reward: 评估奖励
        """
        # 切换到评估模式
        self.policy.set_training_mode(False)
        
        # 重置环境
        obs, _ = eval_env.reset()
        
        # 评估
        done = False
        total_reward = 0
        
        while not done:
            with th.no_grad():
                obs_tensor = obs_as_tensor(obs, self.device)
                actions, _, _ = self.policy(obs_tensor)
                actions = actions.cpu().numpy()
                
            obs, rewards, terminations, truncations, infos = eval_env.step(actions)
            done = terminations[0] or truncations[0]
            total_reward += rewards[0]
            
        # 切换回训练模式
        self.policy.set_training_mode(True)
        
        print(f"global_step={global_step}, eval_reward={total_reward:.2f}")
        
        return total_reward 