""" Treasure hunt evaluation """
from ..treasure_hunt import MultiModalandPose
from ..utils import set_multiple_camera_params
from tesse.msgs import *

from .benchmark import Benchmark


class TreasureHuntBenchmark(Benchmark):
    def __init__(self, config):
        """ Configure evaluation.

        Args:
            config (dict): Evaluation parameters
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
        self.env = MultiModalandPose(
            environment_file=config["environment_file"],
            scene_id=self.scenes[0],
            success_dist=config["success_dist"],
            n_targets=config["n_targets"],
            max_steps=config["episode_length"],
            step_rate=self.STEP_RATE,
            init_hook=set_multiple_camera_params,
            launch_tesse=config["launch_tesse"] if "launch_tesse" in config else True
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
            n_found_targets = 0
            n_predictions = 0
            n_collisions = 0
            step = 0

            if episode > 1:  # scene 1 is set during initialization
                self.env.env.request(SceneRequest(self.scenes[episode]))
            if self.random_seeds:
                self.env.env.request(SetRandomSeed(self.random_seeds[episode]))
            self.env.n_targets = self.n_targets[episode]
            agent.reset()
            obs = self.env.reset()

            for step in range(self.episode_length):
                action = agent.act(obs)
                obs, reward, done, info = self.env.step(action)
                n_found_targets += info["n_found_targets"]

                if action == 3:
                    n_predictions += 1
                if info["collision"]:
                    n_collisions += 1
                if done:
                    break

            results[episode] = {
                "n_found_targets": n_found_targets,
                "precision": n_found_targets / n_predictions,
                "recall": n_found_targets / self.env.n_targets,
                "collisions": n_collisions,
                "n_steps": step + 1,
            }

        return results
