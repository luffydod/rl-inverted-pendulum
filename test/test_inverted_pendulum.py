import pytest
import numpy as np
import sys
import os
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inverted_pendulum import InvertedPendulumEnv
from env.env_factory import make_envs

# 测试直接创建环境和通过工厂函数创建环境
@pytest.fixture
def direct_env():
    return InvertedPendulumEnv(render_mode=None)

@pytest.fixture
def factory_env():
    env = make_envs("inverted-pendulum", 1, render_mode=None)
    return env

# 测试环境初始化
def test_env_initialization(direct_env):
    # 检查观察空间
    assert isinstance(direct_env.observation_space, gym.spaces.Box)
    assert direct_env.observation_space.shape == (2,)  # [alpha, alpha_dot]
    
    # 检查动作空间
    assert isinstance(direct_env.action_space, gym.spaces.Discrete)
    assert direct_env.action_space.n == 3
    
    # 检查参数设置
    assert direct_env.max_episode_steps == 300
    assert direct_env.max_voltage == 3.0
    assert direct_env.discrete_action == True

# 测试连续动作空间
def test_continuous_action_space():
    env = InvertedPendulumEnv(discrete_action=False)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -env.max_voltage
    assert env.action_space.high[0] == env.max_voltage

# 测试离散状态空间
def test_discrete_state_space():
    env = InvertedPendulumEnv(discrete_state=True)
    obs, _ = env.reset()
    # 检查离散观察是否为整数数组
    assert obs.dtype == np.int32
    assert obs.shape == (2,)

# 测试环境重置
def test_env_reset(direct_env):
    obs, info = direct_env.reset()
    # 检查观察的形状和类型
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    
    # 检查观察范围
    assert -np.pi <= obs[0] <= np.pi  # alpha 在 [-π, π] 范围内
    assert -15*np.pi <= obs[1] <= 15*np.pi  # alpha_dot 在 [-15π, 15π] 范围内

# 测试指定初始状态的重置
def test_reset_with_options(direct_env):
    alpha = 0.5
    alpha_dot = 1.0
    obs, _ = direct_env.reset(options={"alpha": alpha, "alpha_dot": alpha_dot})
    np.testing.assert_almost_equal(obs[0], alpha)
    np.testing.assert_almost_equal(obs[1], alpha_dot)

# 测试目标状态的重置
def test_reset_to_goal():
    env = InvertedPendulumEnv(reset_option="goal")
    obs, _ = env.reset()
    np.testing.assert_almost_equal(obs[0], np.pi)  # 目标位置是竖直向上 (alpha = π)
    np.testing.assert_almost_equal(obs[1], 0.0)    # 初始没有角速度

# 测试单步执行
def test_step(direct_env):
    direct_env.reset()
    action = 1  # 中间动作
    
    # 执行一步
    next_obs, reward, terminated, truncated, info = direct_env.step(action)
    
    # 检查返回值
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # 检查状态边界
    assert -np.pi <= next_obs[0] <= np.pi
    assert -15*np.pi <= next_obs[1] <= 15*np.pi

# 测试连续动作执行
def test_continuous_action_step():
    env = InvertedPendulumEnv(discrete_action=False)
    env.reset()
    action = np.array([1.5])  # 连续动作
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # 检查基本返回值
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (2,)
    
    # 检查奖励计算 (负的状态惩罚)
    expected_reward = -(5 * next_obs[0]**2 + 0.1 * next_obs[1]**2 + action[0]**2)
    np.testing.assert_almost_equal(reward, expected_reward)

# 测试多步执行
def test_multiple_steps(direct_env):
    direct_env.reset()
    steps = []
    rewards = []
    
    for _ in range(10):
        action = direct_env.action_space.sample()
        obs, reward, terminated, truncated, _ = direct_env.step(action)
        steps.append(obs)
        rewards.append(reward)
    
    # 检查执行了10步
    assert len(steps) == 10
    assert len(rewards) == 10
    
    # 检查步数计数器
    assert direct_env.steps == 10

# 测试是否正确记录步数和终止条件
def test_termination(direct_env):
    direct_env.reset()
    
    # 执行最大步数
    for _ in range(direct_env.max_episode_steps):
        _, _, terminated, truncated, _ = direct_env.step(1)
        if terminated or truncated:
            break
    
    # 检查是否达到了最大步数并终止
    assert direct_env.steps == direct_env.max_episode_steps
    assert truncated == True  # 因超过最大步数而终止

# 测试向量化环境
def test_vectorized_env(factory_env):
    # 检查向量化环境的观察和动作空间
    assert hasattr(factory_env, 'single_observation_space')
    assert hasattr(factory_env, 'single_action_space')
    
    # 检查重置和步进
    obs, _ = factory_env.reset()
    assert obs.shape[0] == 1  # 批次维度
    
    action = factory_env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = factory_env.step(action)
    
    assert next_obs.shape[0] == 1
    assert reward.shape[0] == 1
    assert terminated.shape[0] == 1
    assert truncated.shape[0] == 1