import numpy as np
import torch
from typing import NamedTuple
from dataclasses import dataclass
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Dict, Any, Union, Optional

class PrioritizedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    
class PrioritizedExperienceReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,    # 优先级指数
        beta: float = 0.4,     # IS权重初始值
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            handle_timeout_termination=handle_timeout_termination,
        )
        
        # PER相关参数
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # 初始化优先级存储
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.max_priority = 1.0

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos: Dict[str, Any],
    ) -> None:
        # 调用父类的add方法添加经验
        super().add(obs, next_obs, action, reward, done, infos)
        
        # 新样本使用最大优先级
        self.priorities[self.pos] = self.max_priority

    def sample(self, batch_size: int):
        # 计算采样概率
        if self.full:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
            
        # 将优先级转换为概率
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # 根据优先级采样索引
        indices = np.random.choice(
            len(probs), 
            size=batch_size, 
            p=probs,
            replace=True
        )
        
        # 计算重要性采样权重
        weights = (len(probs) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取经验数据
        data = super()._get_samples(indices)
        
        return PrioritizedReplayBufferSamples(
            observations=data.observations,
            next_observations=data.next_observations,
            actions=data.actions,
            rewards=data.rewards,
            dones=data.dones,
            weights=torch.FloatTensor(weights).to(self.device),
            indices=torch.LongTensor(indices).to(self.device)
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """更新优先级"""
        priorities = (np.abs(priorities) + self.epsilon) ** self.alpha
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())