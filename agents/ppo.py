from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import flax
import optax
import ml_collections

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


# ---------- small helpers ----------

def _shuffle_minibatches(n: int, minibatch_size: int, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, int]:
    idx = jnp.arange(n)
    idx = jax.random.permutation(key, idx)
    n_minibatches = int(np.ceil(n / minibatch_size))
    return idx, n_minibatches


def _compute_gae(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray,
                 last_value: jnp.ndarray, gamma: float, lam: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages + returns for a *single trajectory* (length T).
    Args:
        rewards: (T,)
        values:  (T,)
        dones:   (T,) with 1.0 where episode ended after that step, else 0.0
        last_value: scalar V(s_{T}) used for bootstrapping (0 if terminal)
    Returns:
        advantages: (T,)
        returns:    (T,)
    """
    T = rewards.shape[0]

    def scan_fn(carry, t):
        lastgaelam = carry
        t_rev = T - 1 - t
        nonterminal = 1.0 - dones[t_rev]
        next_value = jnp.where(t_rev == T - 1, last_value, values[t_rev + 1])
        delta = rewards[t_rev] + gamma * next_value * nonterminal - values[t_rev]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        return lastgaelam, lastgaelam

    _, adv_rev = jax.lax.scan(scan_fn, 0.0, jnp.arange(T))
    adv = jnp.flip(adv_rev, axis=0)
    returns = adv + values
    return adv, returns


@dataclass
class PPOConfig:
    lr: float = 3e-4
    batch_size: int = 2048            # rollout length per update (with 1 env)
    minibatch_size: int = 64
    n_epochs: int = 10
    actor_hidden_dims: tuple = (256, 256)
    value_hidden_dims: tuple = (256, 256)
    layer_norm: bool = False
    actor_layer_norm: bool = False
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    normalize_advantage: bool = True
    actor_fc_scale: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = None


class PPOAgent(flax.struct.PyTreeNode):
    """On-policy PPO agent with GAE and minibatch/epoch updates.

    This version is *norb-compatible*: the public API mirrors ACRLPD so it can be
    plugged into `main_norb.py` and trained online. Use `sample_actions(..., return_log_prob=True)`
    while collecting a rollout of length `config["batch_size"]`, then call `update(...)`
    with the whole rollout (see the updated `main_norb.py`).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    EPS = 1e-6

    # ---- Distribution helpers ----
    # # @jax.jit
    # def _dist_values(self, observations, params=None):
    #     params = params if params is not None else self.network.params
    #     obs = jnp.atleast_2d(observations)
    #     dist = self.network.select("actor")(obs, params=params)
    #     values = self.network.select("value")(obs, params=params)
    #     values = jnp.ravel(values) 
    #     return dist, values

    @partial(jax.jit, static_argnames=("return_log_prob",))
    def sample_actions(self, observations, rng=None, return_log_prob: bool = False):
        """JIT-safe: returns only arrays, never the Distrax object."""
        rng = self.rng if rng is None else rng
        obs = jnp.atleast_2d(observations)
        # build dist inside jit
        dist = self.network.select("actor")(obs)
        if return_log_prob:
            actions, log_prob = dist.sample_and_log_prob(seed=rng)
            return actions.squeeze(0), log_prob.squeeze(0)  # arrays
        else:
            actions = dist.sample(seed=rng)
            return actions.squeeze(0) 

    @jax.jit
    def value(self, observations):
        """JIT-safe value eval, no actor/dist involved."""
        obs = jnp.atleast_2d(observations)
        values = self.network.select("value")(obs)
        values = jnp.ravel(values)
        return jax.lax.cond(values.shape[0] == 1,  # keep this inside JAX
                            lambda v: v[0],
                            lambda v: v,
                            values)

    # ---- One minibatch update ----
    @staticmethod
    @jax.jit
    def _minibatch_update(network: TrainState, batch: Dict[str, jnp.ndarray],
                          clip_eps: float, entropy_coef: float, value_coef: float,
                          normalize_advantage: bool):
        def loss_fn(params):
            obs = jnp.atleast_2d(batch['observations'])
            acts = jnp.atleast_2d(batch['actions'])

            dist = network.select('actor')(obs, params=params)
            values = network.select('value')(obs, params=params)
            values = jnp.ravel(values)

            # make sure actions in (-1, 1) for tanh-gaussian log-prob stability
            safe_actions = jnp.clip(acts, -1 + 1e-6, 1 - 1e-6)
            log_prob = dist.log_prob(safe_actions)

            # importance ratio
            ratio = jnp.exp(jnp.clip(log_prob - batch['old_log_prob'], -20.0, 20.0))

            advantages = batch['advantages']
            norm_flag = jnp.asarray(normalize_advantage)
            def _do_norm(a):
                m = jnp.mean(a)
                s = jnp.std(a)
                s = jnp.where(s > 0, s, 1.0)  # avoid div-by-zero
                return (a - m) / (s + 1e-8)
            advantages = jax.lax.cond(norm_flag, _do_norm, lambda a: a, advantages)

            clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

            v_loss = 0.5 * jnp.mean((batch['returns'] - values) ** 2)
            entropy = -jnp.mean(log_prob)

            total_loss = pg_loss + value_coef * v_loss - entropy_coef * entropy
            info = {
                "pg_loss": pg_loss,
                "v_loss": v_loss,
                "entropy": entropy,
            }
            return total_loss, info

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(network.params)
        new_network = network.apply_gradients(grads=grads)
        return new_network, info

    # ---- Public update API (expects a whole rollout) ----
    @staticmethod
    def _update(self, batch):
        """Perform a full PPO update given a rollout.

        Accepted batch schemas:
        A) {observations, actions, old_log_prob, rewards, dones, last_value}
           - we will compute advantages/returns via GAE
        B) {observations, actions, old_log_prob, advantages, returns}
           - we use provided advantages/returns
        Shapes for A):
           observations: (T, obs_dim), actions: (T, act_dim), rewards/dones/old_log_prob: (T,), last_value: scalar
        """
        new_rng, rng = jax.random.split(self.rng)

        # ---- unify inputs ----
        obs = jnp.asarray(batch['observations'])
        acts = jnp.asarray(batch['actions'])
        old_logp = jnp.asarray(batch['old_log_prob'])

        if 'advantages' in batch and 'returns' in batch:
            advantages = jnp.asarray(batch['advantages'])
            returns = jnp.asarray(batch['returns'])
        else:
            rewards = jnp.asarray(batch['rewards']).astype(jnp.float32)
            # prefer provided 'dones' else derive from 'masks'
            if 'dones' in batch:
                dones = jnp.asarray(batch['dones']).astype(jnp.float32)
            else:
                # masks==1 for non-terminal; so dones = 1 - masks
                dones = 1.0 - jnp.asarray(batch['masks']).astype(jnp.float32)
            # compute values with *current* net (standard in many PPO impls)
            values = self.network.select('value')(jnp.atleast_2d(obs))
            values = jnp.ravel(values)
            last_val = jnp.asarray(batch.get('last_value', 0.0)).astype(jnp.float32)
            advantages, returns = _compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                last_value=last_val,
                gamma=float(self.config['discount']),
                lam=float(self.config['gae_lambda']),
            )

        # flatten to arrays
        T = obs.shape[0]
        # Shuffle once per epoch
        mean_pg, mean_v, mean_ent = 0.0, 0.0, 0.0
        n_steps = 0
        net = self.network

        for epoch in range(int(self.config['n_epochs'])):
            rng, sub = jax.random.split(rng)
            idx, n_minibatches = _shuffle_minibatches(T, int(self.config['minibatch_size']), sub)
            for mb in range(n_minibatches):
                start = mb * int(self.config['minibatch_size'])
                end = min((mb + 1) * int(self.config['minibatch_size']), T)
                mb_idx = idx[start:end]
                mbatch = {
                    'observations': obs[mb_idx],
                    'actions': acts[mb_idx],
                    'old_log_prob': old_logp[mb_idx],
                    'advantages': advantages[mb_idx],
                    'returns': returns[mb_idx],
                }
                net, info = PPOAgent._minibatch_update(
                    network=net,
                    batch=mbatch,
                    clip_eps=float(self.config['clip_eps']),
                    entropy_coef=float(self.config['entropy_coef']),
                    value_coef=float(self.config['value_coef']),
                    normalize_advantage=bool(self.config['normalize_advantage']),
                )
                n_steps += 1
                mean_pg += (info['pg_loss'] - mean_pg) / n_steps
                mean_v += (info['v_loss'] - mean_v) / n_steps
                mean_ent += (info['entropy'] - mean_ent) / n_steps

        # Assign updated network
        self = self.replace(network=net)

        # Post-update diagnostics (approx KL + clip fraction)
        dist_new = self.network.select('actor')(jnp.atleast_2d(obs))
        # keep actions in (-1,1)
        safe_acts = jnp.clip(jnp.atleast_2d(acts), -1 + 1e-6, 1 - 1e-6)
        new_logp = dist_new.log_prob(safe_acts)
        ratio = jnp.exp(jnp.clip(new_logp - old_logp, -20.0, 20.0))
        approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio + 1e-8))
        clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > float(self.config['clip_eps']))

        info = {
            'pg_loss': mean_pg,
            'value_loss': mean_v,
            'entropy': mean_ent,
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
        }
        return self, info

    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # ---- Creation ----
    @classmethod
    def create(cls, seed: int, ex_observations: np.ndarray, ex_actions: np.ndarray, config: Dict[str, Any]):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        actor_def = Actor(
            hidden_dims=tuple(config["actor_hidden_dims"]),
            action_dim=ex_actions.shape[-1],
            layer_norm=bool(config.get("actor_layer_norm", False)),
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=float(config.get("actor_fc_scale", 0.01)),
        )
        value_def = Value(
            hidden_dims=tuple(config["value_hidden_dims"]),
            layer_norm=bool(config.get("layer_norm", False)),
            num_ensembles=1,
        )

        network_def = ModuleDict(dict(actor=actor_def, value=value_def))
        tx = optax.chain(
            optax.clip_by_global_norm(float(config.get("max_grad_norm", 0.5))),
            optax.adam(learning_rate=float(config["lr"]))
        )
        network_params = network_def.init(init_rng, actor=ex_observations, value=ex_observations)["params"]
        network = TrainState.create(network_def, network_params, tx=tx)

        return cls(rng, network=network, config=config)


# Public config for config_flags.DEFINE_config_file

def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="ppo",
            lr=3e-4,
            batch_size=2048,      # rollout length per PPO update
            minibatch_size=64,
            n_epochs=10,
            actor_hidden_dims=(512, 512),
            value_hidden_dims=(512, 512),
            layer_norm=False,
            actor_layer_norm=False,
            discount=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            normalize_advantage=True,
            actor_fc_scale=0.01,
            max_grad_norm=0.5,
            target_kl=None,
        )
    )
