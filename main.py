import time
import sys
import pygame
import argparse
import numpy as np
from rl import train_pendulum, test_pendulum
from env import InvertedPendulumEnv

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test"])
parser.add_argument("-p", "--model_path", type=str, default=None)
args = parser.parse_args()
    
if __name__ == "__main__":
    if args.mode == "train":
        train_pendulum()
    elif args.mode == "test":
        test_pendulum(model_path=args.model_path)