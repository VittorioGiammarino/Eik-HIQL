from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib import patches


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders

def draw(env, ax=None):
    if not ax:
        ax = plt.gca()
    env_u = env.unwrapped
    S = env_u._maze_unit
    for i in range(len(env_u.maze_map)):
        for j in range(len(env_u.maze_map[0])):
            struct = env_u.maze_map[i][j]
            if struct == 1:
                rect = patches.Rectangle((j * S - (env_u._offset_x) - S / 2, i * S - (env_u._offset_y) - S / 2), S, S, 
                                         linewidth=1, edgecolor='none', facecolor='grey', alpha=1.0)
                ax.add_patch(rect)
                

def plot_value_function_grid(agent, 
                            agent_name, 
                            task_id, 
                            env, 
                            grid_size=100, 
                            output_path="value_function.png",
                            draw_maze=True,
                            ):

    _, info = env.reset(options=dict(task_id=task_id, render_goal=False))
    goal = info.get('goal')

    env_u = env.unwrapped
    S = env_u._maze_unit

    x_min, x_max = 0 * S - (env_u._offset_x) - S / 2, len(env_u.maze_map[0]) * S - (env_u._offset_x) - S / 2
    y_min, y_max = 0 * S - (env_u._offset_y) - S / 2, len(env_u.maze_map) * S - (env_u._offset_y) - S / 2
    x_values = np.linspace(x_min, x_max, grid_size)
    y_values = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_values, y_values)
    positions = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Prepare batched input for value_net
    goal_repeated = np.repeat(goal[2:].reshape(1, -1), positions.shape[0], axis=0)
    batch_input = np.hstack([positions, goal_repeated])
    batch_goal = np.repeat(goal.reshape(1, -1), positions.shape[0], axis=0)
    
    # Compute the value function for the grid
    if agent_name in ['qrl']:
        value_net = agent.network.select('value')
        value_function_output = -value_net(batch_input, batch_goal)

    elif agent_name in ['gcivl', 'hiql', 'pi_hiql']:
        value_net = agent.network.select('value')
        v1, v2 = value_net(batch_input, batch_goal)
        value_function_output = (v1 + v2) / 2

    elif agent_name in ['crl', 'gciql']:
        dist = agent.network.select('actor')(batch_input, batch_goal)
        actions = jnp.clip(dist.mode(), -1, 1)
        q1, q2 = agent.network.select('critic')(batch_input, batch_goal, actions)
        value_function_output = (q1 + q2) / 2

    value_function_grid = value_function_output.reshape(grid_size, grid_size)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(value_function_grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='plasma', vmin=-175, vmax=0, zorder=1)
    plt.colorbar(im, ax=ax, label='Value (Goal-Conditioned)')

    # Add contour lines
    contour_levels = np.linspace(value_function_grid.min(), value_function_grid.max(), 50)  # Adjust levels if needed
    ax.contour(X, Y, value_function_grid, levels=contour_levels, colors='black', linewidths=0.5, zorder=2)

    # Overlay the maze structure (with higher zorder)

    if draw_maze:
        draw(env, ax)
        for patch in ax.patches:
            patch.set_zorder(3)  # Ensure maze patches have the highest z-order

    ax.axis('off')

    # Mark start and goal positions
    goal_position = goal[:2]
    ax.plot(goal_position[0], goal_position[1], 'ro', markersize=10, label='Goal', zorder=4)

    ax.set_title("Goal-Conditioned Value Function")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend(loc='lower left')
    plt.savefig(output_path + ".png", format="png", bbox_inches='tight', pad_inches=0)
    plt.savefig(output_path + ".pdf", format="pdf", bbox_inches='tight', pad_inches=0)
    plt.close()
