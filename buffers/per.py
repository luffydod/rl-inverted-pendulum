from typing import Dict, Optional, Union, Any, List, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize

from buffers.buffer import ReplayBufferSamples
from buffers.replay import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.
    
    This buffer samples transitions with probability proportional to their priority.
    The priority is typically based on the TD error of the transition.
    
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
    :param beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
    :param beta_increment: How much to increment beta by on each update
    :param epsilon: Small constant to ensure all transitions have a non-zero probability
    :param max_priority: Maximum priority to assign to new transitions
    :param optimize_memory_usage: Enable a memory efficient variant
    :param handle_timeout_termination: Handle timeout termination separately
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        max_priority: float = 1.0,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = max_priority
        
        # Initialize priorities
        self.priorities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.priorities.fill(self.max_priority)
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add a new transition to the buffer with maximum priority.
        """
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Set maximum priority for the new transition
        if self.pos > 0:
            self.priorities[self.pos - 1] = self.max_priority
        else:
            self.priorities[self.buffer_size - 1] = self.max_priority
            
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> Tuple[ReplayBufferSamples, np.ndarray, np.ndarray]:
        """
        Sample elements from the replay buffer with probability proportional to their priority.
        
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling
        :return: Tuple of (samples, indices, weights)
        """
        if self.optimize_memory_usage:
            # Do not sample the element with index `self.pos` as the transitions is invalid
            if self.full:
                batch_inds = self._sample_proportional(batch_size, exclude_pos=self.pos)
            else:
                batch_inds = self._sample_proportional(batch_size, max_pos=self.pos)
        else:
            if self.full:
                batch_inds = self._sample_proportional(batch_size)
            else:
                batch_inds = self._sample_proportional(batch_size, max_pos=self.pos)
                
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        
        # Calculate importance sampling weights
        weights = self._calculate_weights(batch_inds, env_indices)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return self._get_samples(batch_inds, env=env), batch_inds, weights
    
    def _sample_proportional(self, batch_size: int, exclude_pos: Optional[int] = None, max_pos: Optional[int] = None) -> np.ndarray:
        """
        Sample indices with probability proportional to their priority.
        
        :param batch_size: Number of indices to sample
        :param exclude_pos: Position to exclude from sampling (for memory optimization)
        :param max_pos: Maximum position to sample from
        :return: Array of indices
        """
        if max_pos is None:
            max_pos = self.buffer_size if self.full else self.pos
            
        # Calculate sampling probabilities
        priorities = self.priorities[:max_pos].flatten()
        
        # Exclude the position if specified
        if exclude_pos is not None:
            priorities[exclude_pos] = 0
            
        # Calculate probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(probs), size=batch_size, p=probs)
        
        # Convert back to 2D indices
        env_indices = indices % self.n_envs
        buffer_indices = indices // self.n_envs
        
        return buffer_indices
        
    def _calculate_weights(self, batch_inds: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        """
        Calculate importance sampling weights for the sampled transitions.
        
        :param batch_inds: Indices of the sampled transitions
        :param env_indices: Environment indices of the sampled transitions
        :return: Importance sampling weights
        """
        # Get priorities for the sampled transitions
        priorities = self.priorities[batch_inds, env_indices]
        
        # Calculate weights
        weights = (self.size() * priorities) ** (-self.beta)
        weights /= weights.max()
        
        return weights
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update the priorities of the transitions.
        
        :param indices: Indices of the transitions to update
        :param priorities: New priorities for the transitions
        """
        # Ensure priorities are positive and add epsilon to avoid zero probability
        priorities = np.abs(priorities) + self.epsilon
        
        # Update priorities
        for i, idx in enumerate(indices):
            self.priorities[idx] = priorities[i]
            
        # Update max priority if necessary
        self.max_priority = max(self.max_priority, priorities.max())
        
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Get samples from the buffer.
        
        :param batch_inds: Indices of the samples to get
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling
        :return: Samples from the buffer
        """
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
