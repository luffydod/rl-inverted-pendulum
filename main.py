import argparse
from rl import DQNAgent, DDPGAgent, QLearningAgent

AGENT_MAP = {
    "dqn": DQNAgent,
    "ddqn": DQNAgent,
    "dueling": DQNAgent,
    "ddpg": DDPGAgent,
    "ql": QLearningAgent
}
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", type=str, default="dqn")
parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test"])
parser.add_argument("-p", "--model_path", type=str, default=None)
args = parser.parse_args()
    
if __name__ == "__main__":
    AgentClass = AGENT_MAP.get(args.algorithm)
    if AgentClass is None:
        raise ValueError(f"Algorithm {args.algorithm} not supported")
    agent = AgentClass(algorithm=args.algorithm)
    if args.mode == "train":
        agent.train()
    elif args.mode == "test":
        agent.test(model_path=args.model_path)