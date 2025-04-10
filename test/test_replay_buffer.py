import pytest
import numpy as np
import torch as th
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffers.replay import ReplayBuffer
from env.env_factory import make_envs

N_ENVS = 4
DEVICE = "cpu"
# 使用pytest fixture替代setUp
@pytest.fixture
def env():
    return make_envs("inverted-pendulum", N_ENVS)

@pytest.fixture
def buffer(env):
    buffer_size = 100
    return ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        device=DEVICE,
        n_envs=N_ENVS
    )

@pytest.fixture
def optimized_buffer(env):
    buffer_size = 100
    return ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        device=DEVICE,
        n_envs=N_ENVS,
        optimize_memory_usage=True
    )

def test_buffer_initialization(env, buffer):
    # 测试缓冲区初始化
    assert buffer.buffer_size == 100 // N_ENVS
    assert buffer.pos == 0
    assert not buffer.full
    
    # 检查数组形状
    obs_shape = env.single_observation_space.shape
    assert buffer.observations.shape == (buffer.buffer_size, N_ENVS, *obs_shape)
    assert buffer.next_observations.shape == (buffer.buffer_size, N_ENVS, *obs_shape)
    assert buffer.actions.shape == (buffer.buffer_size, N_ENVS, buffer.action_dim)
    assert buffer.rewards.shape == (buffer.buffer_size, N_ENVS)
    assert buffer.dones.shape == (buffer.buffer_size, N_ENVS)

def test_add_experience(env, buffer):
    # 重置环境
    obs, _ = env.reset()
    
    # 添加一些经验到缓冲区
    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)
        
        buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=[info]
        )
        
        obs = next_obs
    
    # 检查缓冲区状态
    assert buffer.pos == 10
    assert not buffer.full

def test_buffer_fill(env, buffer):
    # 填充缓冲区至满
    obs, _ = env.reset()
    
    for _ in range(buffer.buffer_size + 10):  # 添加比缓冲区容量更多的经验
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        done = np.logical_or(terminated, truncated)
        
        buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=[info]
        )
        
        obs = next_obs
    
    # 检查缓冲区是否已满
    assert buffer.full
    assert buffer.pos == 10  # pos应该已经循环回来

def test_sampling(env, buffer):
    # 先填充缓冲区
    obs, _ = env.reset()
    
    for _ in range(50):  # 添加一些样本
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)
        
        buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=[info]
        )
        
        obs = next_obs
    
    # 从缓冲区采样
    batch_size = 32
    samples = buffer.sample(batch_size=batch_size)
    
    # 验证采样结果
    assert samples.observations.shape[0] == batch_size
    assert samples.actions.shape[0] == batch_size
    assert samples.next_observations.shape[0] == batch_size
    assert samples.dones.shape[0] == batch_size
    assert samples.rewards.shape[0] == batch_size
    
    # 检查数据类型是否为PyTorch张量
    assert isinstance(samples.observations, th.Tensor)
    assert isinstance(samples.actions, th.Tensor)
    assert isinstance(samples.next_observations, th.Tensor)
    assert isinstance(samples.dones, th.Tensor)
    assert isinstance(samples.rewards, th.Tensor)

def test_optimized_buffer(env, optimized_buffer):
    # 测试内存优化版本的缓冲区
    obs, _ = env.reset()
    
    for _ in range(50):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        done = np.logical_or(terminated, truncated)
        
        optimized_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            infos=[info]
        )
        
        obs = next_obs
    
    # 从优化的缓冲区采样
    batch_size = 32
    samples = optimized_buffer.sample(batch_size=batch_size)
    
    # 验证采样结果
    assert samples.observations.shape[0] == batch_size
    assert samples.next_observations.shape[0] == batch_size