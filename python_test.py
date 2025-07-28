import unittest
import time
import logging

class TestUltraFastMADQNSystem(unittest.TestCase):
    """Basic test cases for the UltraFastMADQNSystem"""
    
    def setUp(self):
        """Set up test environment"""
        # Reduce logging output for tests
        logging.getLogger().setLevel(logging.WARNING)
        
        # Initialize system with smaller parameters for faster testing
        self.system = UltraFastMADQNSystem(num_agents=2, grid_size=10)
    
    def test_system_initialization(self):
        """Test that system initializes correctly"""
        self.assertEqual(len(self.system.agents), 2)
        self.assertEqual(self.system.grid_size, 10)
        self.assertEqual(self.system.state_size, 20)  # Based on environment implementation
        self.assertEqual(self.system.action_size, 5)
        
        # Check all agents initialized
        for agent_id, agent in self.system.agents.items():
            self.assertEqual(agent.agent_id, agent_id)
            self.assertEqual(agent.state_size, self.system.state_size)
            self.assertEqual(agent.action_size, self.system.action_size)
    
    def test_episode_execution(self):
        """Test that an episode runs without errors"""
        rewards = self.system.run_episode(training=True)
        
        # Check rewards structure
        self.assertEqual(len(rewards), 2)
        self.assertIn(0, rewards)
        self.assertIn(1, rewards)
        
        # Check performance monitoring
        perf_report = self.system.performance_monitor.get_performance_report()
        self.assertGreater(perf_report['total_decisions'], 0)
    
    def test_training(self):
        """Test that training runs without errors"""
        start_time = time.time()
        self.system.train(num_episodes=5)  # Very short training
        
        # Verify some training occurred
        status = self.system.get_system_status()
        self.assertEqual(status['episode_count'], 5)
        self.assertGreater(sum(status['memory_sizes'].values()), 0)
        
        # Should complete quickly (basic sanity check)
        self.assertLess(time.time() - start_time, 10.0)
    
    def test_model_saving_loading(self):
        """Test model saving and loading"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save models
            self.system.save_models(temp_dir)
            
            # Verify files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "agent_0_model.pth")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "agent_1_model.pth")))
            
            # Create new system and load models
            new_system = UltraFastMADQNSystem(num_agents=2, grid_size=10)
            new_system.load_models(temp_dir)
            
            # Verify models loaded
            status = new_system.get_system_status()
            self.assertEqual(status['agent_epsilons'][0], self.system.agents[0].epsilon)
    
    def test_shutdown(self):
        """Test that shutdown works without errors"""
        self.system.shutdown()
        
        # Verify executor is shutdown
        self.assertTrue(self.system.training_executor._shutdown)

# Run the tests
if __name__ == '__main__':
    # Set up logging for test output
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(TestUltraFastMADQNSystem('test_system_initialization'))
    suite.addTest(TestUltraFastMADQNSystem('test_episode_execution'))
    suite.addTest(TestUltraFastMADQNSystem('test_training'))
    suite.addTest(TestUltraFastMADQNSystem('test_model_saving_loading'))
    suite.addTest(TestUltraFastMADQNSystem('test_shutdown'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
