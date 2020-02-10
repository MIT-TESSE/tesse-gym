""" Treasure hunt evaluation """
from typing import Dict, List, Optional

from ..treasure_hunt import MultiModalandPose
from tesse.msgs import *

from .benchmark import Benchmark
from .agent import Agent


EVALUATION_METRICS = ["found_targets", "precision", "recall", "collisions", "steps"]


class TreasureHuntBenchmark(Benchmark):
    def __init__(
        self,
        scenes: List[int],
        episode_length: List[int],
        n_targets: List[int],
        build_path: str,
        success_dist: int,
        launch_tesse: bool,
        random_seeds: Optional[List[bool]] = None,
    ):
        """ Configure evaluation.

        Args:
            scenes (List[int]): Scene IDs.
            episode_length (List[int]): Episode length.
            n_targets (List[int]): Number of targets.
            build_path (str): Path to TESSE build.
            success_dist (int): Maximum distance from target to be considered found.
            launch_tesse (bool): True to spawn TESSE. Otherwise assume existing instance.
            random_seeds (Optional(List[int])): Optional random seeds for each episode.
        """
        super().__init__()
        self.scenes = scenes
        self.episode_length = episode_length
        self.random_seeds = random_seeds
        self.n_targets = n_targets
        self.env = MultiModalandPose(
            build_path=build_path,
            scene_id=self.scenes[0],
            success_dist=success_dist,
            n_targets=n_targets[0],
            max_steps=max(episode_length),
            step_rate=self.STEP_RATE,
            launch_tesse=launch_tesse,
        )

    def evaluate(self, agent: Agent) -> Dict[str, Dict[str, float]]:
        """ Evaluate agent.

        Args:
            agent (Agent): Agent to be evaluated.

        Returns:
            Dict[str, Dict[str, float]]: Results for each scene
                and an overall performance summary.
        """
        results = {}
        for episode in range(len(self.scenes)):
            print(f"Evaluation episode on scene: {episode}")
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

            for step in range(self.episode_length[episode]):
                action = agent.act(obs)
                obs, reward, done, info = self.env.step(action)
                n_found_targets += info["n_found_targets"]

                if action == 3:
                    n_predictions += 1
                if info["collision"]:
                    n_collisions += 1
                if done:
                    break

            precision = n_found_targets / n_predictions
            recall = n_found_targets / self.env.n_targets
            results[str(episode)] = {
                "found_targets": n_found_targets,
                "precision": precision,
                "recall": recall,
                "collisions": n_collisions,
                "steps": step + 1,
            }

        self.env.close()

        # combine scene results
        results["total"] = {
            metric: sum([scene_result[metric] for scene_result in results.values()])
            for metric in EVALUATION_METRICS
        }

        for metric in ["precision", "recall"]:
            results["total"][metric] /= len(self.scenes)

        return results
