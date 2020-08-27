import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
from aihabitat import run as run

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():
    # run.run_random_agent()
    agent = "ppo_rgb_transfer"
    run.test_agent(agent)

if __name__ == "__main__":
    example()