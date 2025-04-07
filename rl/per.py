import numpy as np
import torch
from typing import NamedTuple, Dict, Any, Union, Optional
from dataclasses import dataclass
from stable_baselines3.common.buffers import ReplayBuffer

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
        beta_increment: float = 0.0005,
        epsilon: float = 1e-5,
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
        
        # 初始化二叉段树用于计算和与最小值
        self.priority_sum = [0 for _ in range(2 * self.buffer_size)]
        self.priority_min = [float('inf') for _ in range(2 * self.buffer_size)]
        
        # 当前最大优先级
        self.max_priority = 1.0

    def _set_priority_min(self, idx, priority_alpha):
        """更新二叉段树中的最小值"""
        # 叶子节点
        idx += self.buffer_size
        self.priority_min[idx] = priority_alpha

        # 沿祖先节点更新树，直到树的根
        while idx >= 2:
            # 获取父节点索引
            idx //= 2
            # 父节点的值是其两个子节点的最小值
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """更新二叉段树中的和"""
        # 叶子节点
        idx += self.buffer_size
        # 设置叶子节点的优先级
        self.priority_sum[idx] = priority

        # 沿祖先节点更新树，直到树的根
        while idx >= 2:
            # 获取父节点索引
            idx //= 2
            # 父节点的值是其两个子节点的和
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        """计算所有优先级的总和"""
        # 根节点保存所有值的和
        return self.priority_sum[1]

    def _min(self):
        """获取最小优先级"""
        # 根节点保存所有值的最小值
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """查找最大的i，使得前i个优先级之和 <= prefix_sum"""
        # 从根节点开始
        idx = 1
        while idx < self.buffer_size:
            # 如果左子树的和大于所需的和
            if self.priority_sum[idx * 2] > prefix_sum:
                # 进入左子树
                idx = 2 * idx
            else:
                # 否则进入右子树并减去左子树的和
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # 我们现在在叶子节点，减去buffer_size得到实际索引
        return idx - self.buffer_size

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos: Dict[str, Any],
    ) -> None:
        # 获取当前位置
        idx = self.pos
        
        # 调用父类的add方法添加经验
        super().add(obs, next_obs, action, reward, done, infos)
        
        # 新样本使用最大优先级
        priority_alpha = self.max_priority ** self.alpha
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def sample(self, batch_size: int):
        """基于优先级采样经验"""
        # 可用样本数
        available_samples = self.pos if not self.full else self.buffer_size
        
        # 初始化样本
        buffer_indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # 获取总优先级
        total_priority = self._sum()
        
        # 计算最小概率
        prob_min = self._min() / total_priority
        # 计算最大权重
        max_weight = (prob_min * available_samples) ** (-self.beta)
        
        # 获取样本索引
        for i in range(batch_size):
            # 随机选择一个优先级前缀和
            p = np.random.random() * total_priority
            # 找到对应的索引
            idx = self.find_prefix_sum_idx(p)
            buffer_indices[i] = idx
            
            # 计算概率
            prob = self.priority_sum[idx + self.buffer_size] / total_priority
            # 计算权重
            weight = (prob * available_samples) ** (-self.beta)
            # 归一化权重
            weights[i] = weight / max_weight
        
        # 增加beta值
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取经验数据
        data = super()._get_samples(buffer_indices)
        
        return PrioritizedReplayBufferSamples(
            observations=data.observations,
            next_observations=data.next_observations,
            actions=data.actions,
            rewards=data.rewards,
            dones=data.dones,
            weights=torch.FloatTensor(weights).to(self.device),
            indices=torch.LongTensor(buffer_indices).to(self.device)
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """更新样本优先级"""
        # 验证输入
        if len(indices) != len(priorities):
            raise ValueError("索引和优先级数组长度不匹配")
            
        if np.any(indices >= self.buffer_size):
            raise IndexError("索引超出缓冲区范围")
        
        for idx, priority in zip(indices, priorities):
            # 添加epsilon确保非零优先级
            priority = max(self.epsilon, priority)
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
            
            # 计算优先级的alpha次方
            priority_alpha = priority ** self.alpha
            
            # 更新树
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)