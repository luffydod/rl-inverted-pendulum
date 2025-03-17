import argparse
from rl import train_pendulum, test_pendulum

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", type=str, default="dqn", choices=["dqn", "ddqn", "dueling"])
parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test"])
parser.add_argument("-p", "--model_path", type=str, default=None)
args = parser.parse_args()
    
if __name__ == "__main__":
    if args.mode == "train":
        train_pendulum(algorithm=args.algorithm)
    elif args.mode == "test":
        test_pendulum(model_path=args.model_path)