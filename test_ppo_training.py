import math
import unittest

import torch

from ppo_training.cases import ThreeByThreeCaseFactory
from ppo_training.encoding import BranchEncoder, EncodingConfig
from ppo_training.environment import DecisionTreeEnv
from ppo_training.networks import BranchScoringActor, StateValueCritic
from ppo_training.trainer import PPOConfig, PPOTrainer


class PPOTrainingTests(unittest.TestCase):
    def setUp(self):
        self.case = ThreeByThreeCaseFactory(randomize=False)()
        self.config = EncodingConfig(n_robots=3, n_resources=9, n_ports=12)

    def test_environment_exposes_only_decision_nodes_and_reward_telescopes(self):
        env = DecisionTreeEnv(self.case.plans)
        node, branches, terminated = env.reset()
        start_cost = node.g
        reward = 0.0
        while not terminated:
            self.assertGreaterEqual(len(branches), 2)
            for child in branches:
                occupied = [value for value in child.U_temp if value is not None]
                self.assertEqual(len(occupied), len(set(occupied)))
            result = env.step(0)
            reward += result.reward
            node, branches, terminated = (
                result.node,
                result.branches,
                result.terminated,
            )
        self.assertAlmostEqual(reward, -(node.g - start_cost))

    def test_encoder_and_variable_branch_actor_shapes(self):
        env = DecisionTreeEnv(self.case.plans)
        node, branches, terminated = env.reset()
        self.assertFalse(terminated)
        encoder = BranchEncoder(self.case.plans, self.config)
        state = encoder.encode_state(node)
        actions = encoder.encode_actions(node, branches)
        actor = BranchScoringActor(encoder.state_dim, encoder.action_dim)
        critic = StateValueCritic(encoder.state_dim)
        logits = actor(state, actions)
        probabilities = torch.softmax(logits, dim=0)
        self.assertEqual(tuple(state.shape), (encoder.state_dim,))
        self.assertEqual(tuple(actions.shape), (len(branches), encoder.action_dim))
        self.assertEqual(tuple(logits.shape), (len(branches),))
        self.assertAlmostEqual(float(probabilities.sum().detach()), 1.0, places=6)
        self.assertEqual(critic(state).ndim, 0)

    def test_one_ppo_update_is_finite(self):
        encoder = BranchEncoder(self.case.plans, self.config)
        actor = BranchScoringActor(encoder.state_dim, encoder.action_dim, hidden_dim=32)
        critic = StateValueCritic(encoder.state_dim, hidden_dim=32)
        trainer = PPOTrainer(
            actor,
            critic,
            encoding_config=self.config,
            config=PPOConfig(
                update_epochs=1,
                minibatch_size=64,
                max_decisions_per_episode=100,
            ),
            seed=7,
        )
        factory = ThreeByThreeCaseFactory(seed=7, randomize=False)
        episodes = trainer.collect_rollouts(factory, episodes=2)
        stats = trainer.update()
        self.assertEqual(len(episodes), 2)
        self.assertGreater(stats.rollout_steps, 0)
        self.assertTrue(math.isfinite(stats.actor_loss))
        self.assertTrue(math.isfinite(stats.critic_loss))
        self.assertTrue(math.isfinite(stats.entropy))


if __name__ == "__main__":
    unittest.main()
