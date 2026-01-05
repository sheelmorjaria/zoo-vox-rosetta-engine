#!/usr/bin/env python3
"""
Test suite for Deep Reinforcement Learning enhancement
TDD implementation for Phase IV feature
"""

import unittest
import sys
import os
sys.path.append('src')

from realtime.deep_reinforcement_learning import (
    DeepReinforcementLearning,
    ReinforcementLearner,
    EnvironmentModel,
    PolicyNetwork,
    ExperienceReplay,
    ExplorationStrategy,
    RewardFunction,
    DQNAgent,
    PPOAgent,
    A2CAgent,
    SACAgent,
    HERAgent,
    MultiAgentLearner,
    MetaLearner,
    SafetyChecker,
    CurriculumScheduler,
    EnsembleLearner
)


class TestDeepReinforcementLearning(unittest.TestCase):
    """Test cases for Deep Reinforcement Learning system"""

    def setUp(self):
        """Set up test environment"""
        self.drl = DeepReinforcementLearning()
        self.test_observation = {'features': [0.1, 0.2, 0.3, 0.4]}
        self.test_action = {'type': 'vocalization', 'parameters': {'frequency': 1000}}
        self.test_reward = 0.5
        self.test_done = False

    def test_learner_creation(self):
        """Test that reinforcement learner can be created"""
        learner = ReinforcementLearner()
        self.assertIsNotNone(learner)
        self.assertIsNotNone(learner.policy_network)
        self.assertIsNotNone(learner.experience_replay)

    def test_environment_model(self):
        """Test environment model functionality"""
        env_model = EnvironmentModel()

        # Test prediction
        prediction = env_model.predict(self.test_observation, self.test_action)
        self.assertIsInstance(prediction, dict)
        self.assertIn('next_state', prediction)
        self.assertIn('reward', prediction)

    def test_policy_network(self):
        """Test policy network operations"""
        policy = PolicyNetwork()

        # Test action selection
        action = policy.select_action(self.test_observation)
        self.assertIsInstance(action, dict)

        # Test policy update
        loss = policy.update_policy([self.test_observation], [self.test_action], [self.test_reward])
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_experience_replay(self):
        """Test experience replay buffer"""
        replay = ExperienceReplay(buffer_size=1000)

        # Add experience
        replay.add_experience(
            observation=self.test_observation,
            action=self.test_action,
            reward=self.test_reward,
            next_observation=self.test_observation,
            done=self.test_done
        )

        # Sample experience
        batch = replay.sample_batch(batch_size=32)
        self.assertEqual(len(batch), 32)

        # Check buffer properties
        self.assertEqual(len(replay), 1)
        self.assertFalse(replay.is_full())

    def test_exploration_strategy(self):
        """Test exploration strategies"""
        # Test epsilon-greedy
        epsilon_greedy = ExplorationStrategy(strategy='epsilon_greedy', epsilon=0.1)
        action = epsilon_greedy.get_action(self.test_observation, lambda: self.test_action)
        self.assertIsInstance(action, dict)

        # Test boltzmann
        boltzmann = ExplorationStrategy(strategy='boltzmann', temperature=1.0)
        action = boltzmann.get_action(self.test_observation, lambda: self.test_action)
        self.assertIsInstance(action, dict)

        # Test ucb
        ucb = ExplorationStrategy(strategy='ucb', c=1.0)
        action = ucb.get_action(self.test_observation, lambda: self.test_action)
        self.assertIsInstance(action, dict)

    def test_reward_function(self):
        """Test reward function calculation"""
        reward_func = RewardFunction()

        # Test basic reward
        reward = reward_func.calculate_reward(
            observation=self.test_observation,
            action=self.test_action,
            next_observation=self.test_observation
        )
        self.assertIsInstance(reward, float)

        # Test shaped reward
        shaped_reward_func = RewardFunction(use_shaping=True)
        reward = shaped_reward_func.calculate_reward(
            observation=self.test_observation,
            action=self.test_action,
            next_observation=self.test_observation
        )
        self.assertIsInstance(reward, float)

    def test_dqn_agent(self):
        """Test DQN agent implementation"""
        agent = DQNAgent()

        # Test learning
        loss = agent.learn(
            observation=self.test_observation,
            action=self.test_action,
            reward=self.test_reward,
            next_observation=self.test_observation,
            done=self.test_done
        )
        self.assertIsInstance(loss, (float, type(None)))

        # Test action selection
        action = agent.act(self.test_observation)
        self.assertIsInstance(action, dict)

        # Test policy saving/loading
        agent.save_policy('test_dqn_policy.pkl')
        self.assertTrue(os.path.exists('test_dqn_policy.pkl'))
        agent.load_policy('test_dqn_policy.pkl')
        os.remove('test_dqn_policy.pkl')

    def test_ppo_agent(self):
        """Test PPO agent implementation"""
        agent = PPOAgent()

        # Test policy update
        trajectories = [
            {'observations': [self.test_observation], 'actions': [self.test_action],
             'rewards': [self.test_reward]}
        ]
        policy_loss, value_loss = agent.update(trajectories)
        self.assertIsInstance(policy_loss, float)
        self.assertIsInstance(value_loss, float)

        # Test action selection
        action = agent.act(self.test_observation)
        self.assertIsInstance(action, dict)

    def test_a2c_agent(self):
        """Test A2C agent implementation"""
        agent = A2CAgent()

        # Test learning step
        loss = agent.step(
            observation=self.test_observation,
            action=self.test_action,
            reward=self.test_reward,
            next_observation=self.test_observation,
            done=self.test_done
        )
        self.assertIsInstance(loss, dict)

        # Test parallel rollout
        rollouts = agent.parallel_rollouts([self.test_observation] * 4)
        self.assertEqual(len(rollouts), 4)

    def test_sac_agent(self):
        """Test SAC agent implementation"""
        agent = SACAgent()

        # Test training
        actor_loss, critic_loss = agent.train(
            observation=self.test_observation,
            action=self.test_action,
            reward=self.test_reward,
            next_observation=self.test_observation,
            done=self.test_done
        )
        self.assertIsInstance(actor_loss, float)
        self.assertIsInstance(critic_loss, float)

        # Test action selection with exploration
        action = agent.act(self.test_observation, explore=True)
        self.assertIsInstance(action, dict)

    def test_her_agent(self):
        """Test HER (Hindsight Experience Replay) agent"""
        agent = HERAgent()

        # Test goal-conditioned learning
        goal = {'target_frequency': 2000}
        goal_conditioned_loss = agent.learn_with_her(
            observation=self.test_observation,
            action=self.test_action,
            next_observation=self.test_observation,
            goal=goal,
            achieved_goal=self.test_observation
        )
        self.assertIsInstance(goal_conditioned_loss, float)

    def test_multi_agent_learner(self):
        """Test multi-agent learning system"""
        multi_agent = MultiAgentLearner(num_agents=3)

        # Test communication
        messages = multi_agent.communicate([0, 1, 2], 'exchange_info')
        self.assertIsInstance(messages, list)

        # Test coordinated action
        coordinated_action = multi_agent.get_coordinated_action([
            self.test_observation, self.test_observation, self.test_observation
        ])
        self.assertIsInstance(coordinated_action, dict)

    def test_meta_learner(self):
        """Test meta-learning capabilities"""
        meta_learner = MetaLearner()

        # Test meta-knowledge update
        meta_knowledge = meta_learner.update_meta_knowledge(
            task_id='task1',
            performance_metrics={'accuracy': 0.8}
        )
        self.assertIsInstance(meta_knowledge, dict)

        # Test fast adaptation
        adapted_policy = meta_learner.fast_adapt(
            new_task_data=[self.test_observation],
            num_adaptation_steps=5
        )
        self.assertIsNotNone(adapted_policy)

    def test_safety_checker(self):
        """Test safety checking system"""
        safety_checker = SafetyChecker()

        # Test safety validation
        is_safe = safety_checker.is_safe_action(
            observation=self.test_observation,
            action=self.test_action
        )
        self.assertIsInstance(is_safe, bool)

        # Test constraint violation detection
        constraints = {
            'min_frequency': 100,
            'max_frequency': 10000,
            'max_amplitude': 1.0
        }
        violation = safety_checker.check_constraints(
            action=self.test_action,
            constraints=constraints
        )
        self.assertIsInstance(violation, list)

    def test_curriculum_scheduler(self):
        """Test curriculum learning scheduler"""
        scheduler = CurriculumScheduler()

        # Test task difficulty adjustment
        difficulty = scheduler.adjust_difficulty(
            performance_metrics={'success_rate': 0.9},
            current_difficulty=0.5
        )
        self.assertIsInstance(difficulty, float)
        self.assertGreaterEqual(difficulty, 0.0)
        self.assertLessEqual(difficulty, 1.0)

        # Test curriculum generation
        curriculum = scheduler.generate_curriculum(
            num_tasks=10,
            complexity_range=(0.1, 1.0)
        )
        self.assertEqual(len(curriculum), 10)

    def test_ensemble_learner(self):
        """Test ensemble learning system"""
        ensemble = EnsembleLearner(num_models=5)

        # Test ensemble prediction
        predictions = ensemble.predict(self.test_observation)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 5)

        # Test ensemble update
        update_loss = ensemble.update_ensemble([
            self.test_observation, self.test_observation
        ], [self.test_action, self.test_action])
        self.assertIsInstance(update_loss, float)

    def test_integration_with_main_system(self):
        """Test integration with main analysis system"""
        # Test that DRL can be integrated with main pipeline
        self.drl.initialize_environment()

        # Test learning loop
        for episode in range(3):
            state = self.drl.reset_environment()
            done = False

            while not done:
                action = self.drl.select_action(state)
                next_state, reward, done = self.drl.step_environment(action)

                # Store experience
                self.drl.store_experience(state, action, reward, next_state, done)

                # Learn from experience
                if self.drl.should_learn():
                    self.drl.learn_from_experience()

                state = next_state

        # Test policy evaluation
        evaluated_policy = self.drl.evaluate_policy(num_episodes=5)
        self.assertIsInstance(evaluated_policy, dict)
        self.assertIn('average_reward', evaluated_policy)

    def test_real_time_learning(self):
        """Test real-time learning capabilities"""
        self.drl.enable_real_time_learning()

        # Simulate real-time learning
        for i in range(100):
            observation = {'features': [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]}
            action = self.drl.real_time_act(observation)
            reward = 0.5 * (i / 100)  # Simulated reward

            self.drl.real_time_learn(observation, action, reward)

        # Check learning progress
        progress = self.drl.get_learning_progress()
        self.assertIsInstance(progress, dict)
        self.assertIn('episodes_completed', progress)

    def test_multi_objective_learning(self):
        """Test multi-objective reinforcement learning"""
        self.drl.setup_multi_objective(
            objectives=['vocalization_accuracy', 'energy_efficiency', 'social_coherence']
        )

        # Test multi-objective learning
        objectives_rewards = {
            'vocalization_accuracy': 0.8,
            'energy_efficiency': 0.6,
            'social_coherence': 0.7
        }

        combined_reward = self.drl.combine_objectives(objectives_rewards)
        self.assertIsInstance(combined_reward, float)
        self.assertGreater(combined_reward, 0.0)

    def test_transfer_learning(self):
        """Test transfer learning capabilities"""
        # Pre-train on source task
        self.drl.pretrain_on_source_task(
            source_data=[self.test_observation] * 100
        )

        # Transfer to target task
        transfer_performance = self.drl.transfer_to_target_task(
            target_data=[self.test_observation] * 50
        )
        self.assertIsInstance(transfer_performance, dict)
        self.assertIn('improvement', transfer_performance)


if __name__ == '__main__':
    unittest.main()