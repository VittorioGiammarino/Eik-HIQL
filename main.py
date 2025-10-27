import json
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
import ogbench
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import generate_obstacle_coordinates, compute_speed_profile, compute_exponential_speed_profile, setup_egl
from utils.evaluation import evaluate, plot_value_function_grid
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Short_experiments', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 100000, 'Number of training steps.') #original 1000000
flags.DEFINE_integer('log_interval', 100, 'Logging interval.') #5000
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.') #100000
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.') #1000000

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('plot_value_function', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('plot_value_function_during_training', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
flags.DEFINE_integer('gpu_id', 0, 'Use this to specify different GPUs')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)

def main(_):

    jax.config.update("jax_default_device", jax.devices()[FLAGS.gpu_id])
    print(f"Using GPU number {FLAGS.gpu_id}")

    #initialize config
    config = FLAGS.agent

    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)

    if config['agent_name'] in ['pi_hiql']:

        if 'actor_loss' in config:
            speed_profile = config['speed_profile']
            projection_bool = config['projection']

            setup_wandb(project=FLAGS.env_name, 
                        group=FLAGS.run_group, 
                        name=exp_name + '_' + config['agent_name'] + '_' + config['actor_loss'] + '_' + speed_profile + f'_projection_{projection_bool}')

            FLAGS.save_dir = os.path.join(FLAGS.save_dir, 
                                        wandb.run.project, 
                                        config['agent_name'], 
                                        config['actor_loss'] + '_' + speed_profile + f'_projection_{projection_bool}' +'/'+exp_name)
            
        else:
            speed_profile = config['speed_profile']
            projection_bool = config['projection']

            setup_wandb(project=FLAGS.env_name, 
                        group=FLAGS.run_group, 
                        name=exp_name + '_' + config['agent_name'] + '_' + speed_profile + f'_projection_{projection_bool}')

            FLAGS.save_dir = os.path.join(FLAGS.save_dir, 
                                        wandb.run.project, 
                                        config['agent_name'], 
                                        speed_profile + f'_projection_{projection_bool}' +'/'+exp_name)


    elif 'actor_loss' in config:
        setup_wandb(project=FLAGS.env_name, 
                    group=FLAGS.run_group, 
                    name=exp_name + '_' + config['agent_name'] + '_' + config['actor_loss'])

        FLAGS.save_dir = os.path.join(FLAGS.save_dir, 
                                    wandb.run.project, 
                                    config['agent_name'], 
                                    config['actor_loss']+'/'+exp_name)

    else:
        setup_wandb(project=FLAGS.env_name, group=FLAGS.run_group, 
                    name=exp_name + '_' + config['agent_name'])

        FLAGS.save_dir = os.path.join(FLAGS.save_dir, 
                                    wandb.run.project, 
                                    config['agent_name'], 
                                    exp_name)

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    setup_egl()
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(FLAGS.env_name, compact_dataset=True)
    env.reset()

    if 'speed_profile' in config:
        env_u = env.unwrapped
        S = env_u._maze_unit
        obstacle_coordinates = generate_obstacle_coordinates(env_u, S, resolution=0.1)

        if config['speed_profile'] == 'linear':        
            speed_train, speed_min = compute_speed_profile(train_dataset['observations'], obstacle_coordinates)
            speed_val, _ = compute_speed_profile(val_dataset['observations'], obstacle_coordinates)

        elif config['speed_profile'] == 'exponential':
            speed_train, speed_min = compute_exponential_speed_profile(train_dataset['observations'], obstacle_coordinates)
            speed_val, _ = compute_exponential_speed_profile(val_dataset['observations'], obstacle_coordinates)

        elif config['speed_profile'] == 'constant':
            speed_train = np.ones((train_dataset['observations'].shape[0],))
            speed_val = np.ones((val_dataset['observations'].shape[0],))

            speed_min = 0.1

        else:
            NotImplementedError

        train_dataset['speed'] = speed_train
        val_dataset['speed'] = speed_val

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    dataset_class = {'GCDataset': GCDataset, 'HGCDataset': HGCDataset}[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])

        if 'use_obstacles' in config:
            if config['use_obstacles'] and obstacle_coordinates.shape[-1] == batch['observations'].shape[-1]:
                #obstacles
                idxs = np.random.randint(len(obstacle_coordinates), size=config['batch_size_obstacles'])
                obs_batch = jax.tree_util.tree_map(lambda arr: arr[idxs], obstacle_coordinates)
                batch['obstacles'] = obs_batch
                batch['speed_obstacles'] = speed_min*np.ones((config['batch_size_obstacles'],))

        #updates
        agent, update_info = agent.update(batch)

        if FLAGS.plot_value_function_during_training>0 and i % 500 == 0:
            if config['agent_name'] not in ['gcbc']:

                if FLAGS.eval_on_cpu:
                    eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
                else:
                    eval_agent = agent
            
                plot_value_function_grid(agent=eval_agent,
                                    agent_name = config['agent_name'],
                                    task_id=1,
                                    env=env,
                                    grid_size=100,
                                    output_path=os.path.join(FLAGS.save_dir, f'value_function_task_{1}_step_{i}'))

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])

                #obstacles
                if 'use_obstacles' in config:
                    if config['use_obstacles'] and obstacle_coordinates.shape[-1] == batch['observations'].shape[-1]:
                        idxs = np.random.randint(len(obstacle_coordinates), size=config['batch_size_obstacles'])
                        obs_batch = jax.tree_util.tree_map(lambda arr: arr[idxs], obstacle_coordinates)
                        val_batch['obstacles'] = obs_batch

                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
                
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)

                if FLAGS.plot_value_function>0:
                    
                    if config['agent_name'] not in ['gcbc']:
                    
                        plot_value_function_grid(agent=eval_agent,
                                            agent_name = config['agent_name'],
                                            task_id=task_id,
                                            env=env,
                                            grid_size=100,
                                            output_path=os.path.join(FLAGS.save_dir, f'value_function_task_{task_id}_step_{i}'))
                        
                        plot_value_function_grid(agent=eval_agent,
                                            agent_name = config['agent_name'],
                                            task_id=task_id,
                                            env=env,
                                            grid_size=100,
                                            output_path=os.path.join(FLAGS.save_dir, f'value_function_task_{task_id}_no_obstacles_step_{i}'),
                                            draw_maze=False)

            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()

if __name__ == '__main__':
    app.run(main)
