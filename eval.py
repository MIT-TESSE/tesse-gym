import argparse
import yaml
from baselines.agents import *  # specific agent is given in agent-config
from tesse_gym.eval.treasure_hunt_benchmark import TreasureHuntBenchmark
from tesse_gym.eval.agent import Agent
from tesse_gym.eval.utils import get_agent_cls
import pprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str)
    parser.add_argument("--agent-config", type=str)
    args = parser.parse_args()

    with open(args.env_config) as f:
        env_args = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.agent_config) as f:
        agent_args = yaml.load(f, Loader=yaml.FullLoader)

    benchmark = TreasureHuntBenchmark(env_args)
    agent = get_agent_cls(agent_args["name"], Agent)(agent_args)
    results = benchmark.evaluate(agent)

    print("------ Environment Configuration -----")
    pprint.pprint(env_args, depth=2)

    print("\n----- Agent Configuration -----")
    pprint.pprint(agent_args, depth=2)

    print("\n----- Per Episode Score -----")
    pprint.pprint(results, depth=2)


if __name__ == "__main__":
    main()
