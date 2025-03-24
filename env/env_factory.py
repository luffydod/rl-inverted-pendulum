from .inverted_pendulum import InvertedPendulumEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.vector import SyncVectorEnv

def make_env(
    env_id: str, 
    render_mode: str = 'rgb_array', 
    max_episode_steps: int = 200,
    discrete_action: bool = True
):
    def thunk():
        if env_id == "inverted-pendulum":
            env = InvertedPendulumEnv(
                max_episode_steps=max_episode_steps,
                discrete_action=discrete_action,
                render_mode=render_mode
            )
        else:
            raise ValueError(f"Invalid environment ID: {env_id}")
        env = RecordEpisodeStatistics(env)
        return env
    return thunk

def make_envs(env_id: str = "inverted-pendulum",
            num_envs: int = 1,
            render_mode: str = 'rgb_array', 
            max_episode_steps: int = 200,
            discrete_action: bool = True):
    return SyncVectorEnv([make_env(env_id, render_mode, max_episode_steps, discrete_action) for _ in range(num_envs)])