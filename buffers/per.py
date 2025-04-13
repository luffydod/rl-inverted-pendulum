from typing import Dict, Optional, Union, Any, List, Tuple

import numpy as np
import torch as th
from gymnasium import spaces

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
        beta_increment: float = 2e-5,
        epsilon: float = 1e-6,
        max_priority: float = 1.0,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
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
            
    def sample(self, batch_size: int) -> Tuple[ReplayBufferSamples, np.ndarray, np.ndarray]:
        """
        Sample elements from the replay buffer with probability proportional to their priority.
        
        :param batch_size: Number of element to sample
        :return: Tuple of (samples, indices, weights)
        """
        if self.full:
            batch_inds, env_inds = self._sample_proportional(batch_size)
        else:
            batch_inds, env_inds = self._sample_proportional(batch_size, max_pos=self.pos)
                
        # Calculate importance sampling weights
        weights = self._calculate_weights(batch_inds, env_inds)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return self._get_samples(batch_inds, env_inds), batch_inds, weights
    
    def _sample_proportional(self, batch_size: int, max_pos: Optional[int] = None) -> np.ndarray:
        """
        Sample indices with probability proportional to their priority.
        
        :param batch_size: Number of indices to sample
        :param max_pos: Maximum position to sample from
        :return: Array of indices
        """
        if max_pos is None:
            max_pos = self.buffer_size
            
        # Calculate sampling probabilities
        priorities = self.priorities[:max_pos].flatten()
            
        # Calculate probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(probs), size=batch_size, p=probs)
        
        # Convert back to 2D indices
        env_indices = indices % self.n_envs
        buffer_indices = indices // self.n_envs
        
        return buffer_indices, env_indices
        
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
        
    def _get_samples(self, batch_inds: np.ndarray, env_inds: np.ndarray) -> ReplayBufferSamples:
        """
        Get samples from the buffer.
        
        :param batch_inds: Indices of the samples to get
        :return: Samples from the buffer
        """
        next_obs = self.next_observations[batch_inds, env_inds, :]
            
        data = (
            self.observations[batch_inds, env_inds, :],
            self.actions[batch_inds, env_inds, :],
            next_obs,
            # Only use dones that are not due to timeouts
            (self.dones[batch_inds, env_inds] * (1 - self.timeouts[batch_inds, env_inds])).reshape(-1, 1),
            self.rewards[batch_inds, env_inds].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
