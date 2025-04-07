import numpy as np
import torch
from typing import NamedTuple, Dict, Any, Union, Optional
from dataclasses import dataclass
from stable_baselines3.common.buffers import ReplayBuffer

class SumTree:
    """实现高效优先级采样的求和树数据结构"""
    def __init__(self, capacity):
        self.capacity = capacity
        # 树的大小为2*capacity-1，前capacity-1个节点是内部节点，后capacity个节点是叶子节点
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        
    def update(self, idx, priority):
        """更新叶子节点优先级并传播更新"""
        # 将索引转换为树中的位置
        tree_idx = idx + self.capacity - 1
        # 计算变化量
        change = priority - self.tree[tree_idx]
        # 更新叶子节点
        self.tree[tree_idx] = priority
        
        # 向上传播变化
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
            
    def add(self, priority):
        """添加新的优先级到树中"""
        # 使用数据指针作为叶子节点索引
        tree_idx = self.data_pointer + self.capacity - 1
        # 更新优先级
        self.update(self.data_pointer, priority)
        # 移动数据指针
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        return tree_idx
    
    def get(self, v):
        """获取优先级累积和为v的叶子节点"""
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # 如果达到叶子节点，则返回
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            # 如果v小于左子树的和，则向左走
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                # 减去左子树的和，向右走
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        # 将树索引转换回数据索引
        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], data_idx
    
    def total_priority(self):
        """返回所有优先级的总和"""
        return self.tree[0]

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
        priority_decay: float = 0.99, # 未被采样的优先级衰减系数
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
        self.priority_decay = priority_decay
        
        # 初始化求和树
        self.sum_tree = SumTree(self.buffer_size)
        self.max_priority = 1.0
        
        # 存储真实叶子节点索引和缓冲区索引的映射
        self.indices_mapping = np.zeros(self.buffer_size, dtype=np.int32)

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
        tree_idx = self.sum_tree.add(self.max_priority)
        # 更新索引映射
        self.indices_mapping[self.pos-1 if self.pos > 0 else self.buffer_size-1] = tree_idx

    def sample(self, batch_size: int):
        """基于优先级采样经验"""
        # 如果缓冲区还没填满，只采样已有的数据
        available_samples = self.pos if not self.full else self.buffer_size
        
        # 分段采样
        segment_size = self.sum_tree.total_priority() / batch_size
        
        # 用于存储采样结果
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        buffer_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # 分段采样
        for i in range(batch_size):
            # 计算当前段的范围
            a = segment_size * i
            b = segment_size * (i + 1)
            
            # 在段内随机采样
            v = np.random.uniform(a, b)
            
            # 从求和树获取样本
            tree_idx, priority, data_idx = self.sum_tree.get(v)
            
            # 存储结果
            tree_indices[i] = tree_idx
            buffer_indices[i] = data_idx
            priorities[i] = priority
        
        # 计算重要性采样权重
        # P(j)为采样概率，样本权重为 (1/N * 1/P(j))^β = (N*P(j))^(-β)
        sampling_probabilities = priorities / self.sum_tree.total_priority()
        weights = (available_samples * sampling_probabilities) ** (-self.beta)
        weights = weights / weights.max()  # 归一化权重
        
        # 更新beta参数 - 使用非线性增长
        self.beta = min(1.0, self.beta + (1.0 - self.beta) * self.beta_increment)
        
        # 衰减未被采样的优先级
        if self.priority_decay < 1.0 and self.full:
            # 创建衰减掩码
            decay_mask = np.ones(self.buffer_size, dtype=bool)
            decay_mask[buffer_indices] = False
            
            # 获取需要衰减的索引
            decay_indices = np.arange(self.buffer_size)[decay_mask]
            
            # 获取对应的树索引
            decay_tree_indices = self.indices_mapping[decay_indices]
            
            # 对每个树索引进行优先级衰减
            for idx in decay_tree_indices:
                original_idx = idx - (self.buffer_size - 1)
                if 0 <= original_idx < self.buffer_size:  # 确保索引有效
                    current_priority = self.sum_tree.tree[idx]
                    new_priority = current_priority * self.priority_decay
                    self.sum_tree.update(original_idx, new_priority)
        
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
        
        # 计算优先级，只添加epsilon，不应用alpha
        raw_priorities = np.abs(priorities) + self.epsilon
        
        # 应用alpha指数转换优先级
        transformed_priorities = raw_priorities ** self.alpha
        
        # 更新求和树中的优先级
        for idx, priority in zip(indices, transformed_priorities):
            # 获取对应的树索引
            tree_idx = self.indices_mapping[idx]
            # 将缓冲区索引转换为叶子节点索引
            leaf_idx = idx if tree_idx == 0 else idx % self.buffer_size
            # 更新优先级
            self.sum_tree.update(leaf_idx, priority)
        
        # 更新最大优先级
        self.max_priority = min(100.0, max(self.max_priority, raw_priorities.max() ** self.alpha))