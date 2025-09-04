from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class PPOAgent(flax.struct.PyTreeNode):
    """Proximal policy optimization (PPO) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    EPS = 1e-6

    def loss(self, batch, grad_params, rng):
        """Compute the PPO loss."""
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        safe_actions = jnp.clip(batch['actions'], -1 + self.EPS, 1 - self.EPS)
        log_prob = dist.log_prob(safe_actions)
        entropy = -dist.log_prob(dist.sample(seed=rng)).mean()

        value = self.network.select('value')(batch['observations'], params=grad_params)
        next_value = self.network.select('value')(batch['next_observations'], params=grad_params)
        
        target = jax.lax.stop_gradient(batch['rewards'] + self.config['discount'] * batch['masks'] * next_value)
        advantage = target - jax.lax.stop_gradient(value)
        # target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_value
        # advantage = target - value
        if self.config['normalize_advantage']:
            adv_mean = jnp.mean(advantage)
            adv_std = jnp.std(advantage) + 1e-8
            advantage = (advantage - adv_mean) / adv_std

        old_log_prob = batch.get('log_probs', log_prob)
        log_ratio = jnp.clip(log_prob - old_log_prob, -20.0, 20.0)
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(ratio, 1.0 - self.config['clip_eps'], 1.0 + self.config['clip_eps'])
        actor_loss = -(jnp.minimum(ratio * advantage, clipped_ratio * advantage)).mean()

        value_loss = 0.5 * ((target - value) ** 2).mean()

        total_loss = (
            actor_loss
            + self.config['value_coef'] * value_loss
            - self.config['entropy_coef'] * entropy
        )

        info = {
            'actor_loss': actor_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'ratio': ratio.mean(),
            'adv': advantage.mean(),
            'v': value.mean(),
        }
        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        loss, info = self.loss(batch, grad_params, rng)
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('return_log_prob',))
    def sample_actions(self, observations, rng=None, temperature=1.0, return_log_prob=False):
        dist = self.network.select('actor')(observations, temperature=temperature)
        if return_log_prob:
            actions, log_prob = dist.sample_and_log_prob(seed=rng)
            return actions, log_prob
        else:
            actions = dist.sample(seed=rng)
            return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]

        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )

        network_info = dict(
            actor=(actor_def, (ex_observations,)),
            value=(value_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='ppo',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=False,
            actor_layer_norm=False,
            discount=0.99,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            normalize_advantage=True,
            actor_fc_scale=0.01,
        )
    )
    return config
