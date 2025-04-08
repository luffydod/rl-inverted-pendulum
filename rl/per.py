import numpy as np
import torch
import math
from typing import NamedTuple, Dict, Any, Union, Optional
from stable_baselines3.common.buffers import ReplayBuffer

class PrioritizedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor

class SumTree:
    """实现优先经验回放所需的求和树结构"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size+1, 2))+1
        self.tree_size = 2**self.tree_level-1
        self.tree = np.zeros(self.tree_size, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def add(self, idx, value):
        """添加新数据到树中"""
        index = self.cursor
        self.cursor = (self.cursor+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

        # 更新树节点
        self.val_update(index, value)

    def get_val(self, index):
        """获取指定索引的值"""
        tree_index = 2**(self.tree_level-1)-1+index
        return self.tree[tree_index]

    def val_update(self, index, value):
        """更新指定索引的值"""
        tree_index = 2**(self.tree_level-1)-1+index
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        """从指定索引开始重建树"""
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex-1)/2)
            self.reconstruct(tindex, diff)
    
    def total(self):
        """返回所有优先级的总和"""
        return self.tree[0]
    
    def find(self, value):
        """查找值对应的索引"""
        return self._find(value, 0)

    def _find(self, value, index):
        """递归查找值对应的索引"""
        if 2**(self.tree_level-1)-1 <= index:
            return index-(2**(self.tree_level-1)-1), self.tree[index]

        left = self.tree[2*index+1]

        if value <= left:
            return self._find(value, 2*index+1)
        else:
            return self._find(value-left, 2*(index+1))
        
    def filled_size(self):
        """返回已填充的大小"""
        return self.size

class PropPrioritizedExperienceReplayBuffer(ReplayBuffer):
    """基于比例的优先经验回放缓冲区"""
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,    # 优先级指数
        beta: float = 0.4,     # IS权重初始值
        beta_annealing_steps: Optional[int] = None,  # beta从初始值到1的步数
        epsilon: float = 1e-5, # 避免优先级为0
        max_priority: float = 1.0,  # 初始最大优先级
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
        self.beta_start = beta
        self.beta = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.epsilon = epsilon
        self.max_priority = max_priority
        self.sample_count = 0  # 用于跟踪采样次数
        
        # 初始化求和树
        self.sum_tree = SumTree(buffer_size)

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos: Dict[str, Any],
    ) -> None:
        """添加新样本到缓冲区"""
        # 获取当前位置
        idx = self.pos
        
        # 调用父类的add方法添加经验
        super().add(obs, next_obs, action, reward, done, infos)
        
        # 新样本使用最大优先级
        priority_alpha = self.max_priority ** self.alpha
        self.sum_tree.add(idx, priority_alpha)

    def sample(self, batch_size: int):
        """基于优先级采样经验"""
        # 可用样本数
        available_samples = min(self.pos, self.buffer_size) if not self.full else self.buffer_size
        
        if available_samples == 0:
            raise ValueError("没有可用的样本进行采样")
        
        # 初始化样本索引和权重数组
        buffer_indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # 获取总优先级
        total_priority = self.sum_tree.total()
        if total_priority <= 0:
            # 处理边缘情况：所有优先级为0
            buffer_indices = np.random.randint(0, available_samples, size=batch_size)
            weights = np.ones_like(buffer_indices, dtype=np.float32)
        else:
            # 计算分段
            segment = total_priority / batch_size
            
            # 计算当前beta值
            if self.beta_annealing_steps:
                progress = min(1.0, self.sample_count / self.beta_annealing_steps)
                self.beta = self.beta_start + progress * (1.0 - self.beta_start)
            
            # 最小概率对应的最大权重
            min_prob = self.epsilon / total_priority
            max_weight = (min_prob * available_samples) ** (-self.beta)
            
            # 采样
            for i in range(batch_size):
                # 在每段内随机选择一个值，确保覆盖整个优先级范围
                a = segment * i
                b = segment * (i + 1)
                p = np.random.uniform(a, b)
                
                # 找到对应的索引
                idx, priority = self.sum_tree.find(p)
                # 确保索引在有效范围内
                idx = min(idx, available_samples - 1)
                buffer_indices[i] = idx
                
                # 计算采样概率
                prob = priority / total_priority
                
                # 计算IS权重
                weight = (prob * available_samples) ** (-self.beta)
                weights[i] = weight / max_weight
        
        self.sample_count += 1
        
        # 获取经验数据
        data = self._get_samples(buffer_indices)
        
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
        if len(indices) == 0:
            return
        
        for idx, priority in zip(indices, priorities):
            # 确保优先级非负且非零
            priority = float(max(self.epsilon, priority))
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
            
            # 计算优先级的alpha次方
            priority_alpha = priority ** self.alpha
            
            # 更新树
            self.sum_tree.val_update(idx, priority_alpha)

class RankPrioritizedExperienceReplayBuffer(ReplayBuffer):
    """基于排名的优先经验回放缓冲区
    
    与基于TD误差的方法相比，基于排名的方法对异常值更加鲁棒，
    优先级基于样本TD误差的排名而非直接使用TD误差值，能够提供更稳定的训练过程。
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,    # 排名优先级指数
        beta: float = 0.4,     # IS权重初始值
        beta_annealing_steps: Optional[int] = None,  # beta从初始值到1的步数
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
        self.beta_start = beta
        self.beta = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.sample_count = 0  # 用于跟踪采样次数
        
        # 存储所有优先级（TD误差）
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        
        # 当优先级被更新时标记为True，用于在下次采样前重新计算排名
        self.priorities_need_update = False
        
        # 存储排名优先级
        self.rank_based_priorities = np.zeros(buffer_size, dtype=np.float32)
        self.sum_priorities = 0.0
        
        # 设置初始优先级为较大值，确保新样本被采样
        self.max_priority = 1.0
        
    def _compute_rank_based_priorities(self):
        """计算基于排名的优先级"""
        # 只计算已填充部分的排名
        filled_size = min(self.pos, self.buffer_size) if not self.full else self.buffer_size
        
        if filled_size == 0:
            return
            
        # 获取优先级数组的前filled_size个元素
        priorities_subset = self.priorities[:filled_size]
        
        # 对优先级进行排序，获取排名（降序排名，即最大的优先级排名为0
        ranks = np.argsort(np.argsort(-priorities_subset))
        
        # 基于排名计算优先级（幂律分布）
        self.rank_based_priorities[:filled_size] = 1.0 / (ranks + 1) ** self.alpha
        if filled_size < self.buffer_size:
            self.rank_based_priorities[filled_size:] = 0.0
            
        # 计算优先级总和
        self.sum_priorities = np.sum(self.rank_based_priorities[:filled_size])
        
        # 标记优先级已更新
        self.priorities_need_update = False
        
    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos: Dict[str, Any],
    ) -> None:
        """添加新样本到缓冲区"""
        # 获取当前位置
        idx = self.pos
        
        # 调用父类的add方法添加经验
        super().add(obs, next_obs, action, reward, done, infos)
        
        # 新样本使用最大优先级
        self.priorities[idx] = self.max_priority
        
        # 标记需要更新排名优先级
        self.priorities_need_update = True
        
    def update_beta(self):
        """更新beta值，用于重要性采样权重计算"""
        if self.beta_annealing_steps is None:
            return
        # 线性增加beta值，从beta_start到1.0
        progress = min(1.0, self.sample_count / self.beta_annealing_steps)
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)
        
    def sample(self, batch_size: int):
        """基于排名优先级采样经验"""
        # 如果优先级需要更新，则重新计算排名优先级
        if self.priorities_need_update:
            self._compute_rank_based_priorities()
            
        # 可用样本数
        available_samples = min(self.pos, self.buffer_size) if not self.full else self.buffer_size
        
        buffer_indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        self.update_beta()
        
        min_prob = self.rank_based_priorities[:available_samples].min() / self.sum_priorities
        max_weight = (min_prob * available_samples) ** (-self.beta)
        
        probs = self.rank_based_priorities[:available_samples] / self.sum_priorities
        for i in range(batch_size):
            # sample
            idx = np.random.choice(
                available_samples, 
                p=probs
            )
            buffer_indices[i] = idx
            
            # IS Weight
            prob = self.rank_based_priorities[idx] / self.sum_priorities
            # w=(p(i)*N)^(-beta)
            weight = (prob * available_samples) ** (-self.beta)
            weights[i] = weight / max_weight
        
        self.sample_count += 1
        
        # 获取经验数据
        data = self._get_samples(buffer_indices)
        
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
        
        if len(indices) == 0:
            return
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
        
        self.priorities_need_update = True