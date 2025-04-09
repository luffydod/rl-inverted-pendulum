# 强化学习作业

## 项目介绍

这是一个使用强化学习方法解决经典场景问题的小项目，本项目环境配置和主要依赖如下：

- 任务环境使用 `gymnasium` 设计，提供了强化学习标准接口和环境模拟
- 经验回放、经验回滚基于 `stable-baselines3` 代码进行删减和定制
- `wandb（Weights & Biases）` 用于训练日志管理和可视化

实现的任务场景：

- 倒立摆✅
- 冰壶游戏✅

当前支持强化学习算法：

- Q-Learning
- DQN
- DDQN
- Dueling DQN
- DDPG
- PPO

补充：实现了优先经验回放（PER）缓冲池。

## 待探讨的问题

- 向量化环境和缓冲池的协同设计

## 运行方式

当前支持两种运行模式：

- 训练模式（train）：训练新模型或继续训练已有模型
- 测试模式（test）：使用训练好的模型进行测试

## 命令行参数

- `-a`, `--algorithm`：指定使用的算法，默认为`dqn`
- `-m`, `--mode`：指定运行模式，可选`train`或`test`，默认为`train`
- `-p`, `--model_path`：指定模型路径，用于加载已有模型或保存新模型，默认为None

## 示例命令

### 训练模型

```bash
# 使用DQN算法训练模型
python main.py -a dqn -m train
python main.py -a dqn

# 使用DDPG算法训练模型
python main.py -a ddpg -m train
python main.py -a ddpg

# 使用PPO算法从已有模型继续训练
python main.py -a ppo -m train -p ./models/ppo_model.pth
python main.py -a ppo -p ./models/ppo_model.pth
```

### 测试模型

```bash
# 使用训练好的DQN模型进行测试
python main.py -a dqn -m test -p ./models/dqn_model.pt

# 使用训练好的Q-Learning模型进行测试
python main.py -a ql -m test -p ./models/ql_model.pt
```

超参数配置，见`./config.py`

## 倒立摆任务描述

![exp](img/exp.png)

### 具体要求

倒立摆系数参数如下表：

| 变量 | 取值        | 单位     | 含义             |
| ---- | ----------- | -------- | ---------------- |
| *m*  | 0.055       | kg       | 重量             |
| *g*  | 9.81        | m/s²     | 重力加速度       |
| *l*  | 0.042       | m        | 重心到转子的距离 |
| *J*  | 1.91 × 10⁻⁴ | kg·m²    | 转动惯量         |
| *b*  | 3 × 10⁻⁶    | Nm·s/rad | 粘滞阻尼         |
| *K*  | 0.0536      | Nm/A     | 转矩常数         |
| *R*  | 9.5         | Ω        | 转子电阻         |

采样时间 $T_s$ 选取0.005s，离散时间动力学 $f$ 可以使用欧拉法获得

$$\alpha_{k+1} = \alpha_k + T_s \dot{\alpha}_k$$

$$\dot{\alpha}_{k+1} = \dot{\alpha}_k + T_s \ddot{\alpha} (\alpha_k, \dot{\alpha}_k, a_k)$$

折扣因子选取 $\gamma=0.98$。选取较高折扣因子的目的是为了提高目标点(顶点)附近奖励在初始时刻状态价值的重要性，这样最优策略能够以成功将摆杆摆起并稳定作为最终目标。

（Tip：可以将动作空间离散化成 {−3,0,3} 三个动作，以这三个动作作为动作集学习最优策略。）

## 倒立摆环境实现

### 经典控制场景

​ `gymnasium` 中实现了经典控制环境倒立摆，可以作为[参考](https://gymnasium.farama.org/environments/classic_control/pendulum/)。

![pendulum](img/pendulum.gif)

​ 值得注意的是，这里是使用超过最大时间步数时截断环境来限制环境的步数。（本来想的是直接根据任务要求通过判断当前状态是否接近[0, 0]来返回是否完成任务。）

```python
def step():
    ...
    # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
    return self._get_obs(), -costs, False, False, {}
```

​ 此外状态重置是通过均匀分布采样，而非每次从初始状态开始，仔细想这样做是增加探索，有利于学习。

```python
self.state = self.np_random.uniform(low=low, high=high)
```

### 状态更新的边界

​ 由于角度范围是 $[-\pi, \pi]$，角速度范围是 $(-15\pi, 15\pi)$，使用欧拉法更新，采样时间 $T_s=0.005$。

$$\theta_{t+1} = \theta_t + \dot\theta_t \cdot T_s$$

​ 考虑倒立摆垂直向下的情况 $\theta_t=\pi$，最大更新量 $\Delta=\dot\theta_t \cdot T_s$，此时有$\pi<\theta_{t+1}<2\pi$，故应取模 $\theta_{t+1}=\theta_{t+1}-2\pi$；

类似的，当 $-2\pi<\theta_{t+1}<-\pi$，应取模 $\theta_{t+1}=\theta_{t+1}+2\pi$。

### wrappers.RecordEpisodeStatistic

此包装器将跟踪累积奖励和剧集时长。

```python
# 示例
info = {
    "episode": {
        "r": "<cumulative reward>",
        "l": "<episode length>",
        "t": "<elapsed time since beginning of episode>"
    },
}
```

## 冰壶游戏任务描述

冰壶游戏是要控制一个半径为1、质量为1的冰壶，在一个长宽是100×100的正方形球场内移动。不考虑冰壶的自转。当冰壶和球场的边界碰撞时，碰撞前后冰壶的速度会乘上回弹系数0.9，移动方向和边界呈反射关系。

我们需要分别操纵x轴和y轴的两个力控制冰壶的移动：

- 在x轴的正或反方向施加5单位的力
- 在y轴的正或反方向施加5单位的力
这样一共会有4种不同的控制动作。

动作可以每 `1/10` 秒变换一次，但在仿真冰壶运动动力学时，仿真时间间隔是 `1/100` 秒。除了我们施加的控制动作，冰壶会受到空气阻力，大小等于 `0.005×speed²`。假设冰壶和地面没有摩擦力。

在每个决策时刻( `1/10` 秒)，环境反馈的奖励等于 `-d`，其中 `d` 是冰壶和任意给定的目标点之间的距离。为了保证学习的策略能够控制冰壶从任意初始位置上移动到任意目标点，每隔 `30` 秒就把冰壶状态重置到球场内的一个随机点上，同时 `x` 轴和 `y` 轴的速度也随机重置在 `[-10, 10]` 范围内。与此同时，目标点也被随机重置。
(提示：把每隔 `30` 秒当成一次轨迹，把问题定义成 `episodic MDPs` 问题，`episodic length = 30/0.1 = 300 steps`。`γ = 1`或`γ = 0.9`)

### 速度边界讨论

只考虑x方向上的运动，冰壶初始位置在左边界，给一个正向最大初速度，每个时间步模拟施加的动作（力）与当前速度等向，在300轮次的模拟中记录最大时刻速度，结果是`29.79`（可能存在误差），因此状态空间速度边界设置为`[-30, 30]`。

### 摩擦力模拟

摩擦力注意是和速度反向的，因此注意计算的时候摩檫力`f = - mu * v * |v|`。

```python
# cal force
f = -self.mu * self.v * np.abs(self.v) + f0
```

## DQN

近似值：

$$Q(S_{t}, a_t; \theta_t)$$

```python
# Q(s_t, a_t)
proximate_values = q_network(data.observations).gather(1, data.actions).squeeze()
```

目标值：

$$Y_t^{\text{DQN}} = R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a; \theta_t^-)$$
对应代码实现

```python
# data.
#  rewards: (batch_size, 1)
#  observations, next_observations: (batch_size, state_dim)
# actions: (batch_size, action_dim)
# target_max: (batch_size, )
#  (b, s)- - Q` - ->(b, a) - - max - -> val(b, ), indice(b, )
target_max, _ = target_network(data.next_observations).max(dim=1)

# (b, )
target_values = data.rewards.flatten() + config["gamma"] * target_max * (1 - data.dones.flatten())
            
```

## DDQN

目标值：

$$Y_t^{\text{DoubleDQN}} = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_{a} Q(S_{t+1}, a; \theta_t), \theta_t^-)$$
对应代码实现：

```python
# (b, s) - - Q - -> (b, a) - - argmax - -> (b, 1)
max_q_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)

# Q`(s_{t+1}, max_q_actions)
target_values = data.rewards.flatten() \
                        + config["gamma"] \
                        * target_network(data.next_observations).gather(1, max_actions).squeeze() \
                        * (1 - data.dones.flatten())
```

## Dueling DQN

和原始DQN相比，将Q值分离为状态值和动作优势估计。

（1）减去最大值，最优动作具有零优势

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \max_{a' \in |\mathcal{A}|} A(s, a'; \theta, \alpha) \right)$$

（2）使用均值替代

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)$$

网络架构对比：

![duelingdqn](img/duelingdqn.png)

对应源码实现：

```python
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=3):
        super().__init__()
        self.embedding_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.v_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.a_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        embedding = self.embedding_network(x)
        v = self.v_network(embedding)
        a = self.a_network(embedding)
        return v + (a - a.mean(dim=1, keepdim=True))
```

## 对比实验

### tau参数

​ 使用Dueling Double DQN训练倒立摆，实验固定参数如下：

```json
batch_size:128
buffer_size:2,000,000
end_epsilon:0.05
eval_frequency:10,000
exploration_fraction:0.2
gamma:0.98
learning_rate:0.0001
learning_starts:5,000
max_grad_norm:10
n_envs:8
start_epsilon:1
target_network_frequency:1,000
total_timesteps:1,000,000
train_frequency:4
```

​ 变量控制：目标网络更新系数 $\tau$，按照 $\theta_{\text{target}} \leftarrow \tau \theta_{\text{current}} + (1-\tau)\theta_{\text{target}}$ 更新。实验组为$\tau=1.0,0.8,0.6,0.4,0.2,0.1$。

​ TD Loss 随 $\tau$ 减小而下移，$\tau$ 越小，目标网络更新越"温和"，导致 `td_loss` 降低；软更新（$\tau <1$）通过逐步混合参数，使目标网络的变化更平滑，减少 Q 值目标的波动性，从而降低时序差分误差；硬更新（$\tau=1$）可能导致目标网络参数突变，使 Q 值目标不稳定，产生更大的 `td_loss`。

![tau_loss](img/tau_loss.png)

![tau_q](img/tau_q.png)

​ 从累计期望奖励走势以及单步奖励走势来看，$\tau=0.4$ 最好。

![tau_train_return](img/tau_train_return.png)

![tau_eval_return](img/tau_eval_return.png)

![tau_reward](img/tau_reward.png)

效果展示：
![play1](img/show_1.gif)

## wandb使用

### wandb.Video

​ 输入的data可以是numpy  array，Channels should be (time, channel, height, width) or (batch, time, channel, height width)，被AI骗了，一直传的`(T, H, W, C)`，我说怎么百试不灵……
