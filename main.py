import argparse
from rl import DQNAgent

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", type=str, default="dqn", choices=["dqn", "ddqn", "dueling"])
parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test"])
parser.add_argument("-p", "--model_path", type=str, default=None)
args = parser.parse_args()
    
if __name__ == "__main__":
    if args.mode == "train":
        agent = DQNAgent(algorithm=args.algorithm)
        agent.train()
    elif args.mode == "test":
        agent = DQNAgent(algorithm=args.algorithm)
        agent.test(model_path=args.model_path)