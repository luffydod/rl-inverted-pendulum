from env.inverted_pendulum import InvertedPendulumEnv
from env.curling import CurlingEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from gymnasium.vector import SyncVectorEnv

def make_env(
    env_id: str, 
    render_mode: str = 'rgb_array',
    discrete_action: bool = True,
    discrete_state: bool = False,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    reset_option: str = None
):
    """创建单个环境实例的工厂函数
    
    参数:
        env_id: 环境ID, 指定要创建的环境类型
        render_mode: 渲染模式, 如'rgb_array'或'human'
        discrete_action: 是否使用离散动作空间
        discrete_state: 是否使用离散状态空间
        normalize_obs: 是否标准化观测值
        normalize_reward: 是否标准化奖励
        reset_option: 环境重置选项
        
    返回:
        thunk: 一个可调用对象，每次调用时创建一个新的环境实例
    """
    def thunk():
        if env_id == "inverted-pendulum":
            env = InvertedPendulumEnv(
                render_mode=render_mode,
                discrete_action=discrete_action,
                discrete_state=discrete_state,
                reset_option=reset_option
            )
        elif env_id == "curling":
            env = CurlingEnv(
                render_mode=render_mode,
                discrete_state=discrete_state
            )
        else:
            raise ValueError(f"Invalid environment ID: {env_id}")
        env = RecordEpisodeStatistics(env)
        if normalize_obs:
            env = NormalizeObservation(env)
        if normalize_reward:
            env = NormalizeReward(env)
        return env
    return thunk

def make_envs(env_id: str = "inverted-pendulum",
            num_envs: int = 1,
            render_mode: str = 'rgb_array',
            discrete_action: bool = True,
            discrete_state: bool = False,
            normalize_obs: bool = False,
            normalize_reward: bool = False,
            reset_option: str = None):
    """创建多个向量化环境实例
    
    参数:
        env_id: 环境ID
        num_envs: 要创建的环境实例数量
        render_mode: 渲染模式
        discrete_action: 是否使用离散动作空间
        discrete_state: 是否使用离散状态空间
        normalize_obs: 是否标准化观测值
        normalize_reward: 是否标准化奖励
        reset_option: 环境重置选项
        
    返回:
        SyncVectorEnv: 包含多个环境实例的向量化环境
    """
    return SyncVectorEnv([make_env(env_id, 
                                   render_mode, 
                                   discrete_action, 
                                   discrete_state, 
                                   normalize_obs,
                                   normalize_reward,
                                   reset_option) for _ in range(num_envs)])