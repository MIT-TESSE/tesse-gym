""" Treasure hunt evaluation """
from ..treasure_hunt import RGBSegDepthInput
from tesse.msgs import *

from .benchmark import Benchmark


class TreasureHuntBenchmark(Benchmark):
    def __init__(self, config):
        """ Configure evaluation.

        Args:
            config (dict): Evaluation parameteros
                - episode_length
                - scenes
                - random seeds (optional)
                - environment file
                - success_dist
                - n_targets
                - step_rate
        """
        super().__init__()
        self.scenes = config["scenes"]
        self.episode_length = config["episode_length"]
        self.random_seeds = config["seeds"] if "seeds" in config else None
        self.n_targets = config["n_targets"]
        self.n_targets = (
            len(self.scenes) * [self.n_targets]
            if isinstance(self.n_targets, int)
            else self.n_targets
        )
        self.env = RGBSegDepthInput(
            environment_file=config["environment_file"],
            scene_id=self.scenes[0],
            success_dist=config["success_dist"],
            n_targets=config["n_targets"],
            max_steps=config["episode_length"],
            step_rate=self.STEP_RATE,
        )

    def evaluate(self, agent):
        """ Evaluate agent over specified configurations.

        Args:
            agent (tesse_gym.eval.agents.Agent): Agent to be evaluated.

        Returns:
            int: Agent's score.
        """
        results = {}
        for episode in range(len(self.scenes)):
            if episode > 0:  # scene 0 is set during initialization
                self.env.env.request(SceneRequest(self.scenes[episode]))
            if self.random_seeds:
                self.env.env.request(SetRandomSeed(self.random_seeds[episode]))
            self.env.n_targets = self.n_targets[episode]
            agent.reset()
            obs = self.env.reset()

            for step in range(self.episode_length):
                action = agent.act(obs)
                obs, reward, done, info = self.env.step(action)

                if done:
                    break

            results[episode] = self.env.n_found_targets

        return results
