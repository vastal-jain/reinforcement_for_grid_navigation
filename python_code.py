"""
Multi-Agent Deep Q-Network (MA-DQN) System for Robotics Navigation
Author: Chief AI Developer
Date: July 26, 2025
Version: 1.0

This module implements a multi-agent reinforcement learning system for robotic navigation
in dynamic environments with unpredictable obstacles and shifting goals.

Features:
- MA-DQN algorithm implementation
- ROS2 compatibility
- Real-time decision making (<60ms)
- Sensor data loss handling
- Live retraining capabilities
- Comprehensive logging system
- Modular and secure design
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import threading
import time
import logging
import json
import hashlib
import queue
from collections import deque, namedtuple
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('madqn_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security configuration
class SecurityManager:
    """Handles security aspects of the system"""
    
    @staticmethod
    def hash_state(state: np.ndarray) -> str:
        """Create secure hash of state for logging"""
        return hashlib.sha256(state.tobytes()).hexdigest()[:16]
    
    @staticmethod
    def validate_action(action: int, action_space_size: int) -> bool:
        """Validate action is within acceptable range"""
        return 0 <= action < action_space_size

# Data structures
@dataclass
class StateActionPair:
    """Container for state-action pairs with metadata"""
    timestamp: float
    agent_id: int
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray]
    done: bool
    sensor_status: Dict[str, bool]
    state_hash: str

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network for individual agents"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience replay buffer with priority sampling"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001
        self.epsilon = 1e-6
    
    def push(self, experience: Experience, priority: float = None):
        """Add experience to buffer"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with priority-based sampling"""
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])
        
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        return len(self.buffer)

class GridEnvironment:
    """25x25 grid environment with dynamic obstacles and goals"""
    
    def __init__(self, grid_size: int = 25, num_agents: int = 4):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.grid = np.zeros((grid_size, grid_size))
        self.agent_positions = {}
        self.goals = {}
        self.obstacles = set()
        self.dynamic_obstacles = set()
        self.last_obstacle_update = time.time()
        self.obstacle_update_interval = 2.0  # seconds
        
        self.reset()
    
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial states"""
        self.grid.fill(0)
        self.obstacles.clear()
        self.dynamic_obstacles.clear()
        
        # Place agents randomly
        positions = random.sample([(i, j) for i in range(self.grid_size) 
                                 for j in range(self.grid_size)], self.num_agents * 2)
        
        for i in range(self.num_agents):
            self.agent_positions[i] = positions[i]
            self.goals[i] = positions[i + self.num_agents]
        
        # Add initial obstacles
        self._generate_obstacles()
        
        return {i: self._get_state(i) for i in range(self.num_agents)}
    
    def _generate_obstacles(self):
        """Generate random obstacles"""
        num_obstacles = random.randint(50, 100)
        occupied_positions = set(self.agent_positions.values()) | set(self.goals.values())
        
        for _ in range(num_obstacles):
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in occupied_positions:
                self.obstacles.add(pos)
                if random.random() < 0.3:  # 30% chance of being dynamic
                    self.dynamic_obstacles.add(pos)
    
    def _update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles"""
        current_time = time.time()
        if current_time - self.last_obstacle_update > self.obstacle_update_interval:
            new_dynamic = set()
            for obs in self.dynamic_obstacles:
                # Move obstacle randomly
                moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
                dx, dy = random.choice(moves)
                new_pos = (max(0, min(self.grid_size-1, obs[0] + dx)),
                          max(0, min(self.grid_size-1, obs[1] + dy)))
                
                # Check if new position is valid
                if (new_pos not in self.agent_positions.values() and 
                    new_pos not in self.goals.values()):
                    new_dynamic.add(new_pos)
                    self.obstacles.discard(obs)
                    self.obstacles.add(new_pos)
                else:
                    new_dynamic.add(obs)
            
            self.dynamic_obstacles = new_dynamic
            self.last_obstacle_update = current_time
    
    def _get_state(self, agent_id: int) -> np.ndarray:
        """Get state representation for agent"""
        pos = self.agent_positions[agent_id]
        goal = self.goals[agent_id]
        
        # Local view (7x7 around agent)
        view_size = 7
        half_view = view_size // 2
        local_view = np.zeros((view_size, view_size, 4))  # 4 channels
        
        for i in range(view_size):
            for j in range(view_size):
                world_i = pos[0] - half_view + i
                world_j = pos[1] - half_view + j
                
                if 0 <= world_i < self.grid_size and 0 <= world_j < self.grid_size:
                    # Channel 0: obstacles
                    if (world_i, world_j) in self.obstacles:
                        local_view[i, j, 0] = 1
                    
                    # Channel 1: other agents
                    for other_id, other_pos in self.agent_positions.items():
                        if other_id != agent_id and other_pos == (world_i, world_j):
                            local_view[i, j, 1] = 1
                    
                    # Channel 2: goals
                    if (world_i, world_j) == goal:
                        local_view[i, j, 2] = 1
                    
                    # Channel 3: agent itself
                    if (world_i, world_j) == pos:
                        local_view[i, j, 3] = 1
        
        # Add global information
        global_info = np.array([
            pos[0] / self.grid_size,
            pos[1] / self.grid_size,
            goal[0] / self.grid_size,
            goal[1] / self.grid_size,
            len(self.obstacles) / (self.grid_size * self.grid_size)
        ])
        
        return np.concatenate([local_view.flatten(), global_info])
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict]:
        """Execute actions for all agents"""
        self._update_dynamic_obstacles()
        
        moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]  # stay, right, left, down, up
        rewards = {}
        dones = {}
        new_positions = {}
        
        # Calculate new positions
        for agent_id, action in actions.items():
            if agent_id in self.agent_positions:
                pos = self.agent_positions[agent_id]
                dx, dy = moves[action]
                new_pos = (max(0, min(self.grid_size-1, pos[0] + dx)),
                          max(0, min(self.grid_size-1, pos[1] + dy)))
                new_positions[agent_id] = new_pos
        
        # Check for collisions and update positions
        for agent_id, new_pos in new_positions.items():
            old_pos = self.agent_positions[agent_id]
            goal = self.goals[agent_id]
            
            # Check for obstacles
            if new_pos in self.obstacles:
                rewards[agent_id] = -10  # Collision penalty
                new_pos = old_pos  # Stay in place
            else:
                # Check for agent collisions
                collision = False
                for other_id, other_new_pos in new_positions.items():
                    if other_id != agent_id and other_new_pos == new_pos:
                        collision = True
                        break
                
                if collision:
                    rewards[agent_id] = -5  # Agent collision penalty
                    new_pos = old_pos
                else:
                    # Calculate reward
                    old_dist = np.linalg.norm(np.array(old_pos) - np.array(goal))
                    new_dist = np.linalg.norm(np.array(new_pos) - np.array(goal))
                    
                    if new_pos == goal:
                        rewards[agent_id] = 100  # Goal reached
                        dones[agent_id] = True
                        # Shift goal
                        self.goals[agent_id] = (random.randint(0, self.grid_size-1),
                                              random.randint(0, self.grid_size-1))
                    else:
                        rewards[agent_id] = (old_dist - new_dist) * 10  # Progress reward
                        dones[agent_id] = False
            
            self.agent_positions[agent_id] = new_pos
        
        # Get new states
        next_states = {i: self._get_state(i) for i in range(self.num_agents)}
        
        return next_states, rewards, dones, {}

class SensorSimulator:
    """Simulates sensor data loss"""
    
    def __init__(self, loss_probability: float = 0.1):
        self.loss_probability = loss_probability
        self.sensor_types = ['lidar', 'camera', 'imu', 'encoder']
    
    def get_sensor_status(self) -> Dict[str, bool]:
        """Get current sensor status"""
        return {sensor: random.random() > self.loss_probability 
                for sensor in self.sensor_types}
    
    def apply_sensor_loss(self, state: np.ndarray, sensor_status: Dict[str, bool]) -> np.ndarray:
        """Apply sensor loss to state"""
        corrupted_state = state.copy()
        
        if not sensor_status.get('lidar', True):
            # Corrupt obstacle information
            corrupted_state[:49] *= 0.5  # Reduce obstacle channel
        
        if not sensor_status.get('camera', True):
            # Corrupt visual information
            corrupted_state[49:98] *= 0.3  # Reduce agent detection
        
        return corrupted_state

class MADQNAgent:
    """Multi-Agent DQN Agent"""
    
    def __init__(self, agent_id: int, state_size: int, action_size: int, 
                 lr: float = 0.001, device: str = 'cpu'):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_frequency = 100
        self.steps = 0
        
        # Replay buffer
        self.memory = ReplayBuffer(10000)
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        
        # Calculate TD error for priority
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            current_q = self.q_network(state_tensor)[0, action]
            next_q = self.target_network(next_state_tensor).max(1)[0]
            target_q = reward + (self.gamma * next_q * (1 - done))
            
            td_error = abs(current_q - target_q).item()
        
        self.memory.push(experience, td_error)
    
    def train(self, batch_size: int = 32) -> float:
        """Train the agent"""
        if len(self.memory) < batch_size:
            return 0.0
        
        experiences, indices, weights = self.memory.sample(batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate loss with importance sampling weights
        td_errors = F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        loss = (td_errors * weights_tensor).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        new_priorities = td_errors.detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_frequency == 0:
            self.update_target_network()
        
        return loss.item()

class DataLogger:
    """Comprehensive logging system"""
    
    def __init__(self, log_file: str = "madqn_data.json"):
        self.log_file = log_file
        self.data_queue = queue.Queue()
        self.security_manager = SecurityManager()
        self.logging_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.logging_thread.start()
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    
    def log_state_action_pair(self, sap: StateActionPair):
        """Log state-action pair"""
        self.data_queue.put(sap)
    
    def _logging_worker(self):
        """Background worker for logging"""
        while True:
            try:
                sap = self.data_queue.get(timeout=1)
                self._write_to_file(sap)
                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Logging error: {e}")
    
    def _write_to_file(self, sap: StateActionPair):
        """Write data to file"""
        data = {
            'session_id': self.session_id,
            'timestamp': sap.timestamp,
            'agent_id': sap.agent_id,
            'action': sap.action,
            'reward': sap.reward,
            'done': sap.done,
            'sensor_status': sap.sensor_status,
            'state_hash': sap.state_hash
        }
        
        try:
            with open(self.log_file, 'a') as f:
                json.dump(data, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"File write error: {e}")

class ROS2Interface:
    """ROS2 compatibility interface"""
    
    def __init__(self):
        self.node_name = "madqn_navigation_node"
        self.publishers = {}
        self.subscribers = {}
        logger.info(f"ROS2 Interface initialized for node: {self.node_name}")
    
    def publish_action(self, agent_id: int, action: int):
        """Publish action command (ROS2 compatible)"""
        # In real implementation, this would publish to ROS2 topics
        logger.debug(f"Publishing action {action} for agent {agent_id}")
    
    def subscribe_sensor_data(self, callback):
        """Subscribe to sensor data (ROS2 compatible)"""
        # In real implementation, this would subscribe to ROS2 topics
        logger.debug("Subscribed to sensor data")

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.decision_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=100)
        self.training_losses = {}
    
    def record_decision_time(self, duration: float):
        """Record decision time"""
        self.decision_times.append(duration)
    
    def get_average_decision_time(self) -> float:
        """Get average decision time"""
        return np.mean(self.decision_times) if self.decision_times else 0.0
    
    def record_training_loss(self, agent_id: int, loss: float):
        """Record training loss"""
        if agent_id not in self.training_losses:
            self.training_losses[agent_id] = deque(maxlen=100)
        self.training_losses[agent_id].append(loss)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        return {
            'avg_decision_time_ms': self.get_average_decision_time() * 1000,
            'max_decision_time_ms': max(self.decision_times) * 1000 if self.decision_times else 0,
            'training_losses': {agent_id: np.mean(losses) 
                              for agent_id, losses in self.training_losses.items()},
            'total_decisions': len(self.decision_times)
        }

class MADQNSystem:
    """Main Multi-Agent DQN System"""
    
    def __init__(self, num_agents: int = 4, grid_size: int = 25):
        self.num_agents = num_agents
        self.grid_size = grid_size
        
        # Initialize components
        self.environment = GridEnvironment(grid_size, num_agents)
        self.sensor_simulator = SensorSimulator()
        self.data_logger = DataLogger()
        self.ros2_interface = ROS2Interface()
        self.performance_monitor = PerformanceMonitor()
        self.security_manager = SecurityManager()
        
        # Get state size from environment
        sample_state = self.environment._get_state(0)
        self.state_size = len(sample_state)
        self.action_size = 5  # stay, right, left, down, up
        
        # Initialize agents
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agents = {i: MADQNAgent(i, self.state_size, self.action_size, device=device)
                      for i in range(num_agents)}
        
        # Training parameters
        self.training_active = True
        self.episode_count = 0
        self.max_episode_steps = 200
        
        # Threading for real-time operation
        self.training_executor = ThreadPoolExecutor(max_workers=num_agents)
        
        logger.info(f"MA-DQN System initialized with {num_agents} agents")
    
    def run_episode(self, training: bool = True) -> Dict[str, float]:
        """Run a single episode"""
        states = self.environment.reset()
        episode_rewards = {i: 0 for i in range(self.num_agents)}
        episode_steps = 0
        
        while episode_steps < self.max_episode_steps:
            start_time = time.time()
            
            # Get sensor status
            sensor_status = self.sensor_simulator.get_sensor_status()
            
            # Select actions for all agents
            actions = {}
            for agent_id in range(self.num_agents):
                state = states[agent_id]
                
                # Apply sensor loss
                corrupted_state = self.sensor_simulator.apply_sensor_loss(state, sensor_status)
                
                # Select action
                action = self.agents[agent_id].select_action(corrupted_state, training)
                
                # Validate action
                if not self.security_manager.validate_action(action, self.action_size):
                    action = 0  # Default to "stay" action
                
                actions[agent_id] = action
                
                # Log state-action pair
                sap = StateActionPair(
                    timestamp=time.time(),
                    agent_id=agent_id,
                    state=state,
                    action=action,
                    reward=0,  # Will be updated after step
                    next_state=None,  # Will be updated after step
                    done=False,  # Will be updated after step
                    sensor_status=sensor_status,
                    state_hash=self.security_manager.hash_state(state)
                )
                self.data_logger.log_state_action_pair(sap)
            
            # Execute actions
            next_states, rewards, dones, _ = self.environment.step(actions)
            
            # Store experiences and train
            if training:
                training_futures = []
                for agent_id in range(self.num_agents):
                    state = states[agent_id]
                    action = actions[agent_id]
                    reward = rewards.get(agent_id, 0)
                    next_state = next_states[agent_id]
                    done = dones.get(agent_id, False)
                    
                    # Store experience
                    self.agents[agent_id].store_experience(state, action, reward, next_state, done)
                    
                    # Submit training task
                    future = self.training_executor.submit(self.agents[agent_id].train)
                    training_futures.append((agent_id, future))
                    
                    episode_rewards[agent_id] += reward
                
                # Collect training results
                for agent_id, future in training_futures:
                    try:
                        loss = future.result(timeout=0.05)  # 50ms timeout
                        if loss > 0:
                            self.performance_monitor.record_training_loss(agent_id, loss)
                    except Exception as e:
                        logger.warning(f"Training timeout for agent {agent_id}: {e}")
            
            # Update states
            states = next_states
            episode_steps += 1
            
            # Record decision time
            decision_time = time.time() - start_time
            self.performance_monitor.record_decision_time(decision_time)
            
            # Check if decision time exceeds 60ms
            if decision_time > 0.060:
                logger.warning(f"Decision time exceeded 60ms: {decision_time*1000:.2f}ms")
            
            # Publish actions to ROS2
            for agent_id, action in actions.items():
                self.ros2_interface.publish_action(agent_id, action)
            
            # Check if all agents are done
            if all(dones.get(i, False) for i in range(self.num_agents)):
                break
        
        self.episode_count += 1
        return episode_rewards
    
    def train(self, num_episodes: int = 1000):
        """Train the system"""
        logger.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            episode_rewards = self.run_episode(training=True)
            
            if episode % 100 == 0:
                avg_reward = np.mean(list(episode_rewards.values()))
                performance_report = self.performance_monitor.get_performance_report()
                
                logger.info(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                          f"Avg Decision Time: {performance_report['avg_decision_time_ms']:.2f}ms")
        
        logger.info("Training completed")
    
    def run_live_simulation(self, duration_seconds: int = 300):
        """Run live simulation with real-time retraining"""
        logger.info(f"Starting live simulation for {duration_seconds} seconds")
        
        start_time = time.time()
        episode_count = 0
        
        while time.time() - start_time < duration_seconds:
            episode_rewards = self.run_episode(training=True)
            episode_count += 1
            
            # Periodic performance reporting
            if episode_count % 10 == 0:
                performance_report = self.performance_monitor.get_performance_report()
                logger.info(f"Live simulation - Episode {episode_count}: "
                          f"Avg Decision Time: {performance_report['avg_decision_time_ms']:.2f}ms")
        
        logger.info(f"Live simulation completed. Ran {episode_count} episodes")
    
    def save_models(self, directory: str = "models"):
        """Save trained models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            model_path = os.path.join(directory, f"agent_{agent_id}_model.pth")
            torch.save({
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'steps': agent.steps
            }, model_path)
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str = "models"):
        """Load trained models"""
        import os
        
        for agent_id, agent in self.agents.items():
            model_path = os.path.join(directory, f"agent_{agent_id}_model.pth")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                agent.epsilon = checkpoint['epsilon']
                agent.steps = checkpoint['steps']
        
        logger.info(f"Models loaded from {directory}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        performance_report = self.performance_monitor.get_performance_report()
        
        return {
            'episode_count': self.episode_count,
            'training_active': self.training_active,
            'performance': performance_report,
            'agent_epsilons': {i: agent.epsilon for i, agent in self.agents.items()},
            'memory_sizes': {i: len(agent.memory) for i, agent in self.agents.items()}
        }

def main():
    """Main function to demonstrate the system"""
    # Initialize system
    system = MADQNSystem(num_agents=4, grid_size=25)
    
    try:
        # Training phase
        logger.info("Starting training phase...")
        system.train(num_episodes=500)
        
        # Save models
        system.save_models()
        
        # Live simulation phase
        logger.info("Starting live simulation phase...")
        system.run_live_simulation(duration_seconds=60)
        
        # System status
        status = system.get_system_status()
        logger.info(f"Final system status: {status}")
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        logger.info("System shutdown")

if __name__ == "__main__":
    main()
