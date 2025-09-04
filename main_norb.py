import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np


FLAGS = flags.FLAGS

if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]

# ---- Core run/setup flags ----
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 0, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# NOTE: pass --agent=agents/ppo.py on CLI to switch to PPO
config_flags.DEFINE_config_file('agent', 'agents/acrlpd.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")


class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f"{prefix}/{k}": float(v) for k, v in data.items()}, step=step)


def _process_reward(env_name: str, r: float) -> float:
    if "antmaze" in env_name and (
        "diverse" in env_name or "play" in env_name or "umaze" in env_name
    ):
        return r - 1.0
    if is_robomimic_env(env_name):
        return r - 1.0
    return r


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project="qc", group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # ---- Env setup ----
    env, eval_env, _, _ = make_env_and_datasets(FLAGS.env_name)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    rng = jax.random.PRNGKey(FLAGS.seed)

    # Example batch for network init
    example_observations, _ = env.reset()
    example_actions = env.action_space.sample()

    # ---- Agent setup ----
    config = FLAGS.agent
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(FLAGS.seed, example_observations, example_actions, config)

    # ---- Logging ----
    prefixes = ["train", "eval", "env"]
    logger = LoggingHelper(
        csv_loggers={
            prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv"))
            for prefix in prefixes
        },
        wandb_logger=__import__("wandb"),
    )

    obs, _ = env.reset()
    global_step = 0
    action_dim = example_actions.shape[-1]

    pbar = tqdm.tqdm(total=FLAGS.online_steps, desc="PPO")

    # ----- Simple on-policy rollout buffer (PPO only) -----
    is_ppo = (config['agent_name'] == 'ppo')
    ppo_rollout = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'old_log_prob': [],
    }

    def maybe_ppo_update(next_obs, done_flag):
        nonlocal agent
        if not is_ppo:
            return
        # trigger when we have enough steps or at episode end if buffer is big
        if len(ppo_rollout['rewards']) == 0:
            return
        reached_batch = len(ppo_rollout['rewards']) >= int(config['batch_size'])
        if (not reached_batch) and (not done_flag):
            return
        # use 0 bootstrap if terminal, else V(s_T)
        last_v = float(agent.value(next_obs)) if not done_flag else 0.0
        batch = {
            'observations': np.asarray(ppo_rollout['observations'], dtype=np.float32),
            'actions': np.asarray(ppo_rollout['actions'], dtype=np.float32),
            'rewards': np.asarray(ppo_rollout['rewards'], dtype=np.float32),
            'dones': np.asarray(ppo_rollout['dones'], dtype=np.float32),
            'old_log_prob': np.asarray(ppo_rollout['old_log_prob'], dtype=np.float32),
            'last_value': np.asarray(last_v, dtype=np.float32),
        }
        agent, train_info = agent.update(batch)
        logger.log(train_info, "train", step=global_step)
        # clear buffer
        for k in ppo_rollout.keys():
            ppo_rollout[k] = []

    while global_step < FLAGS.online_steps:
        rng, key = jax.random.split(rng)
        prev_obs = obs

        if global_step < FLAGS.start_training:
            # warmup random policy
            action = jax.random.uniform(key, shape=(action_dim,), minval=-1, maxval=1)
            log_prob = 0.0
        else:
            if is_ppo:
                action, log_prob = agent.sample_actions(prev_obs, rng=key, return_log_prob=True)
            else:
                action = agent.sample_actions(prev_obs, rng=key)
                log_prob = None
        action = np.asarray(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        # env metrics
        env_info = {k: v for k, v in info.items() if k.startswith("distance")}
        if env_info:
            logger.log(env_info, "env", step=global_step)

        # reward shaping for specific envs
        reward = _process_reward(FLAGS.env_name, float(reward))

        # ---- PPO rollout collection ----
        if is_ppo:
            ppo_rollout['observations'].append(np.asarray(prev_obs, dtype=np.float32))
            ppo_rollout['actions'].append(np.asarray(action, dtype=np.float32))
            ppo_rollout['rewards'].append(np.asarray(reward, dtype=np.float32))
            ppo_rollout['dones'].append(np.asarray(float(done), dtype=np.float32))
            ppo_rollout['old_log_prob'].append(np.asarray(log_prob if log_prob is not None else 0.0, dtype=np.float32))

        # ---- ACRLPD style per-step update (non-PPO) ----
        if (not is_ppo) and (global_step >= FLAGS.start_training):
            batch = {
                'observations': prev_obs,
                'actions': action,
                'rewards': reward,
                'masks': 1.0 - float(terminated),
                'next_observations': next_obs,
                'log_probs': log_prob if log_prob is not None else 0.0,
            }
            agent, train_info = agent.update(batch)
            logger.log(train_info, "train", step=global_step)

        global_step += 1
        pbar.update(1)

        # trigger PPO update if ready *before* resetting obs
        if is_ppo and (global_step >= FLAGS.start_training):
            maybe_ppo_update(next_obs, done)

        # step obs
        obs = next_obs if not done else env.reset()[0]

        # --- Eval & save ---
        if FLAGS.eval_interval > 0 and global_step % FLAGS.eval_interval == 0:
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=global_step)

        if FLAGS.save_interval > 0 and global_step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, global_step)

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    with open(os.path.join(FLAGS.save_dir, "token.tk"), "w") as f:
        f.write(run.url)


if __name__ == "__main__":
    app.run(main)
