import collections
import os
import platform
import time

import gymnasium
import numpy as np
import jax
from scipy.spatial import cKDTree
from gymnasium.spaces import Box

import ogbench
from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.get_observation(), reward, terminated, truncated, info


def setup_egl():
    """Set up EGL for rendering."""
    if 'mac' in platform.platform():
        # macOS doesn't support EGL.
        pass
    else:
        os.environ['MUJOCO_GL'] = 'egl'
        if 'SLURM_STEP_GPUS' in os.environ:
            os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']


def make_env_and_datasets(dataset_name, frame_stack=None):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    setup_egl()

    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()

    return env, train_dataset, val_dataset

def generate_obstacle_coordinates(env_u, S, resolution=0.01):
    """
    Generate a batch of obstacle coordinates based on the environment maze map.

    Args:
        env_u: Environment object containing the maze map and offsets.
        S: Size of each cell in the maze map.
        resolution: Accuracy of the coordinates (e.g., 0.01).

    Returns:
        np.ndarray: A 2D array of obstacle coordinates with shape (N, 2).
    """
    obstacle_coordinates = []

    for i in range(len(env_u.maze_map)):
        for j in range(len(env_u.maze_map[0])):
            struct = env_u.maze_map[i][j]
            if struct == 1:  # Cell is an obstacle
                # Bottom-left corner of the cell
                x_min = j * S - env_u._offset_x - S / 2
                y_min = i * S - env_u._offset_y - S / 2

                # Generate points within the cell with given resolution
                x_values = np.arange(x_min, x_min + S, resolution)
                y_values = np.arange(y_min, y_min + S, resolution)
                X, Y = np.meshgrid(x_values, y_values)
                cell_coordinates = np.stack([X.ravel(), Y.ravel()], axis=1)
                
                # Append to the obstacle coordinates list
                obstacle_coordinates.append(cell_coordinates)
    
    # Combine all coordinates into a single array
    if obstacle_coordinates:
        obstacle_coordinates = np.vstack(obstacle_coordinates)
    else:
        obstacle_coordinates = np.empty((0, 2))  # No obstacles
    
    return obstacle_coordinates

def compute_closest_distance(point, obstacle_coords):
    """
    Compute the distance from a given point to the closest obstacle point.

    Args:
        point (tuple or np.ndarray): The query point (x, y).
        obstacle_coords (np.ndarray): Array of obstacle coordinates with shape (N, 2).

    Returns:
        float: The distance to the closest obstacle point.
    """
    # Build a k-d tree for the obstacle coordinates
    tree = cKDTree(obstacle_coords)

    # Query the distance to the closest point
    distance, _ = tree.query(point)
    return distance

def compute_speed_profile(query_points, obstacle_coords):
    """
    Compute the optimal speed profile for a set of query points.

    Args:
        query_points (np.ndarray): Array of query points with shape (N, 2).
        obstacle_coords (np.ndarray): Array of obstacle coordinates with shape (M, 2).

    Returns:
        np.ndarray: Speed profile for each query point.
    """
    # Build k-d tree for obstacle coordinates
    tree = cKDTree(obstacle_coords)

    # Query distances to closest obstacle
    distances, _ = tree.query(query_points[:, 0:2])
    d_min = np.min(distances)
    d_max = np.max(distances)

    # Compute speed profile
    speed_profile = np.clip(distances/d_max, d_min/d_max, 1.0)
    speed_min = d_min/d_max

    return speed_profile, speed_min

def compute_exponential_speed_profile(query_points, obstacle_coords, speed_min=0.1, decay_rate=1.0):
    """
    Compute the optimal speed profile for a set of query points.

    Args:
        query_points (np.ndarray): Array of query points with shape (N, 2).
        obstacle_coords (np.ndarray): Array of obstacle coordinates with shape (M, 2).
        speed_min (float): Minimum speed value (default: 0.1).
        decay_rate (float): Decay rate controlling the steepness (default: 1.0).

    Returns:
        np.ndarray: Speed profile for each query point.
    """
    # Build k-d tree for obstacle coordinates
    tree = cKDTree(obstacle_coords)

    # Query distances to closest obstacle
    distances, _ = tree.query(query_points[:, 0:2])
    d_min = np.min(distances)
    d_max = np.max(distances)

    assert d_min < d_max, "d_min must be less than d_max"
    assert 0 < speed_min < 1, "speed_min must be in the range (0, 1)"

    # Normalize distances and compute exponential decay
    normalized_distances = (d_max - distances) / (d_max - d_min)
    
    # Compute speed profile
    speed_profile = speed_min + (1 - speed_min) * np.exp(-decay_rate * normalized_distances)
    
    # Clip speeds to ensure they stay within [speed_min, 1]
    speed_profile = np.clip(speed_profile, speed_min, 1.0)

    return speed_profile, speed_min
