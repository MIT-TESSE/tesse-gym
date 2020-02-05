""" Treasure hunt evaluation """
from collections import namedtuple

from ..treasure_hunt import MultiModalandPose
from ..utils import set_multiple_camera_params
from tesse.msgs import *

from .benchmark import Benchmark


EVALUATION_METRICS = ["found_targets", "precision", "recall", "collisions", "steps"]
SceneResults = namedtuple("SceneResults", EVALUATION_METRICS)


def combine_scene_results(scene_results):
    """ Combine benchmark results across several scenes

    Args:
        scene_results (dict[str, SceneResults]):
    """
    results = {}
    for metric in EVALUATION_METRICS:
        results[metric] = sum(
            [getattr(scene_result, metric) for scene_result in scene_results.values()]
        )
        if metric in ["precision", "recall"]:
            results[metric] /= len(scene_results)
    return SceneResults(**results)


class TreasureHuntBenchmark(Benchmark):
    def __init__(
        self,
        scenes,
        episode_length,
        n_targets,
        environment_file,
        success_dist,
        launch_tesse,
        random_seeds=None,
    ):
        """ Configure evaluation.

        Args:
            scenes (List[int]): Scene IDs.
            episode_length (int): Episode length.
            n_targets (int): Number of targets.
            environment_file (str): Path to TESSE build.
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
            environment_file=environment_file,
            scene_id=self.scenes[0],
            success_dist=success_dist,
            n_targets=n_targets,
            max_steps=episode_length,
            step_rate=self.STEP_RATE,
            init_hook=set_multiple_camera_params,
            launch_tesse=launch_tesse,
        )

    def evaluate(self, agent):
        """ Evaluate agent.

        Args:
            agent (tesse_gym.eval.agent.Agent): Agent to be evaluated.

        Returns:
            int: Agent's score.
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

            precision = n_found_targets / n_predictions
            recall = n_found_targets / self.env.n_targets
            results[episode] = SceneResults(
                n_found_targets, precision, recall, n_collisions, step + 1
            )

        self.env.close()
        results["total"] = combine_scene_results(results)
        return results
