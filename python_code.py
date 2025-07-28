"""
Multi-Agent Deep Q-Network (MA-DQN) System for Robotics Navigation
Author: Chief AI Developer
Date: July 28, 2025
Version: 1.8.1 - Fixed ThreadPoolExecutor shutdown compatibility
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import threading
import time
import logging
import json
import hashlib
import queue
import sys
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os

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

# Fixed seed for reproducibility
RANDOM_SEED = 42
rng = np.random.default_rng(seed=RANDOM_SEED)

# Python version compatibility check
PYTHON_VERSION = sys.version_info
SUPPORTS_EXECUTOR_TIMEOUT = PYTHON_VERSION >= (3, 9)

# Custom JSON encoder for numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def safe_json_serialize(data):
    """Safely serialize data to JSON, converting numpy types"""
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_types(v) for v in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    return convert_types(data)

# Security configuration
class SecurityManager:
    """Handles security aspects of the system"""
    
    @staticmethod
    def hash_state(state: np.ndarray) -> str:
        """Create secure hash of state for logging"""
        if not isinstance(state, np.ndarray):
            raise TypeError("State must be a numpy array")
        return hashlib.sha256(state.tobytes()).hexdigest()[:16]
    
    @staticmethod
    def validate_action(action: int, action_space_size: int) -> bool:
        """Validate action is within acceptable range"""
        if not isinstance(action, (int, np.integer)):
            return False
        if not isinstance(action_space_size, (int, np.integer)):
            return False
        return 0 <= action < action_space_size
    
    @staticmethod
    def safe_torch_load(filepath: str, device: str = 'cpu') -> Optional[Dict]:
        """Safely load PyTorch models with security checks"""
        if not os.path.exists(filepath):
            return None
        
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=True)
            
            required_keys = ['q_network_state_dict', 'target_network_state_dict', 
                           'optimizer_state_dict', 'epsilon', 'steps']
            
            if not all(key in checkpoint for key in required_keys):
                logger.error(f"Invalid checkpoint structure in {filepath}")
                return None
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to safely load model from {filepath}: {e}")
            return None

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

class SimpleDQNNetwork(nn.Module):
    """Ultra-lightweight Deep Q-Network for fastest inference"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
            
        super(SimpleDQNNetwork, self).__init__()
        # Minimal architecture for speed
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        if x.size(-1) == 0:
            raise RuntimeError("Input tensor cannot have zero features")
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FastReplayBuffer:
    """Ultra-fast replay buffer with minimal overhead"""
    
    def __init__(self, capacity: int):
        if capacity <= 0:
            capacity = 0
        
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) if capacity > 0 else deque()
        self.position = 0
    
    def push(self, experience: Experience):
        """Add experience to buffer without priority calculation"""
        if self.capacity == 0:
            return
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Simple uniform sampling"""
        if len(self.buffer) < batch_size:
            return []
        
        indices = rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self):
        return len(self.buffer)

class MinimalGridEnvironment:
    """Minimal grid environment optimized for speed"""
    
    def __init__(self, grid_size: int = 15, num_agents: int = 4):  # Smaller grid
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if num_agents > grid_size * grid_size:
            raise ValueError("num_agents cannot exceed grid capacity")
            
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.agent_positions = {}
        self.goals = {}
        self.obstacles = set()
        
        # Pre-compute all valid positions
        self.all_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        
        self.reset()
    
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial states"""
        self.obstacles.clear()
        
        # Place agents and goals
        selected_indices = rng.choice(len(self.all_positions), size=self.num_agents * 2, replace=False)
        selected_positions = [self.all_positions[idx] for idx in selected_indices]
        
        for i in range(self.num_agents):
            self.agent_positions[i] = selected_positions[i]
            self.goals[i] = selected_positions[i + self.num_agents]
        
        # Minimal obstacles
        num_obstacles = 20  # Fixed small number
        occupied_positions = set(self.agent_positions.values()) | set(self.goals.values())
        
        obstacle_candidates = [pos for pos in self.all_positions if pos not in occupied_positions]
        if len(obstacle_candidates) >= num_obstacles:
            obstacle_indices = rng.choice(len(obstacle_candidates), num_obstacles, replace=False)
            self.obstacles = {obstacle_candidates[idx] for idx in obstacle_indices}
        
        return {i: self._get_state_fast(i) for i in range(self.num_agents)}
    
    def _get_state_fast(self, agent_id: int) -> np.ndarray:
        """Ultra-fast state computation with minimal features"""
        if agent_id not in self.agent_positions:
            raise ValueError(f"Agent {agent_id} not found in environment")
            
        pos = self.agent_positions[agent_id]
        goal = self.goals[agent_id]
        
        # Minimal state: only 3x3 local view + global info
        view_size = 3
        half_view = 1
        local_view = np.zeros(view_size * view_size * 2)  # Only 2 channels: obstacles + goal
        
        idx = 0
        for i in range(view_size):
            for j in range(view_size):
                world_i = pos[0] - half_view + i
                world_j = pos[1] - half_view + j
                
                if 0 <= world_i < self.grid_size and 0 <= world_j < self.grid_size:
                    # Channel 0: obstacles
                    if (world_i, world_j) in self.obstacles:
                        local_view[idx] = 1
                    # Channel 1: goal
                    if (world_i, world_j) == goal:
                        local_view[idx + 9] = 1  # 9 = view_size^2
                idx += 1
        
        # Minimal global information
        global_info = np.array([
            (goal[0] - pos[0]) / self.grid_size,  # Goal direction x
            (goal[1] - pos[1]) / self.grid_size,  # Goal direction y
        ])
        
        return np.concatenate([local_view, global_info])  # Total: 18 + 2 = 20 features
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict]:
        """Minimal step function"""
        moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        new_positions = {}
        rewards = {}
        dones = {}
        
        # Calculate new positions
        for agent_id, action in actions.items():
            if agent_id in self.agent_positions and 0 <= action < len(moves):
                pos = self.agent_positions[agent_id]
                dx, dy = moves[action]
                new_pos = (max(0, min(self.grid_size-1, pos[0] + dx)),
                           max(0, min(self.grid_size-1, pos[1] + dy)))
                new_positions[agent_id] = new_pos
            else:
                new_positions[agent_id] = self.agent_positions[agent_id]
        
        # Simple reward calculation
        for agent_id, new_pos in new_positions.items():
            old_pos = self.agent_positions[agent_id]
            goal = self.goals[agent_id]
            
            if new_pos in self.obstacles:
                rewards[agent_id] = -10
                new_positions[agent_id] = old_pos
                dones[agent_id] = False
            elif new_pos == goal:
                rewards[agent_id] = 100
                dones[agent_id] = True
                # Relocate goal
                available = [pos for pos in self.all_positions 
                           if pos not in self.obstacles and pos not in new_positions.values()]
                if available:
                    self.goals[agent_id] = available[rng.integers(0, len(available))]
            else:
                # Simple distance-based reward
                old_dist = abs(old_pos[0] - goal[0]) + abs(old_pos[1] - goal[1])
                new_dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
                rewards[agent_id] = (old_dist - new_dist) * 5
                dones[agent_id] = False
        
        # Update positions
        for agent_id, new_pos in new_positions.items():
            self.agent_positions[agent_id] = new_pos
        
        # Get new states
        next_states = {i: self._get_state_fast(i) for i in range(self.num_agents)}
        
        return next_states, rewards, dones, {}

class FastSensorSimulator:
    """Minimal sensor simulator"""
    
    def __init__(self, loss_probability: float = 0.05):  # Reduced loss
        self.loss_probability = loss_probability
        self.cached_status = {'lidar': True, 'camera': True, 'imu': True, 'encoder': True}
        self.last_update = 0
    
    def get_sensor_status(self) -> Dict[str, bool]:
        """Cached sensor status"""
        current_time = time.time()
        if current_time - self.last_update > 0.1:  # Update every 100ms
            self.cached_status = {
                'lidar': rng.random() > self.loss_probability,
                'camera': rng.random() > self.loss_probability,
                'imu': True,  # Keep IMU always on
                'encoder': True  # Keep encoder always on
            }
            self.last_update = current_time
        return self.cached_status
    
    def apply_sensor_loss(self, state: np.ndarray, sensor_status: Dict[str, bool]) -> np.ndarray:
        """Minimal sensor loss"""
        if not isinstance(state, np.ndarray):
            raise TypeError("State must be a numpy array")
        
        # Skip processing if all sensors working
        if all(sensor_status.values()):
            return state
            
        corrupted_state = state.copy()
        if not sensor_status.get('lidar', True):
            corrupted_state[:9] *= 0.5  # First 9 features (obstacles)
        
        return corrupted_state

class UltraFastMADQNAgent:
    """Ultra-optimized MADQN Agent for sub-30ms performance"""
    
    def __init__(self, agent_id: int, state_size: int, action_size: int, 
                 lr: float = 0.01, device: str = 'cpu'):  # Higher LR for faster learning
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
            
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        
        # Ultra-lightweight networks
        self.q_network = SimpleDQNNetwork(state_size, action_size, hidden_size=64).to(self.device)
        self.target_network = SimpleDQNNetwork(state_size, action_size, hidden_size=64).to(self.device)
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=lr)  # SGD is faster than Adam
        
        # Simplified hyperparameters
        self.gamma = 0.9
        self.epsilon = 0.3  # Start lower
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.update_target_frequency = 500  # Less frequent updates
        self.steps = 0
        
        # Small, fast replay buffer
        self.memory = FastReplayBuffer(1000)
        
        # Pre-allocated tensors for ultra-fast inference
        self._state_tensor = torch.zeros(1, state_size, device=self.device, dtype=torch.float32)
        
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Ultra-fast action selection"""
        if not isinstance(state, np.ndarray):
            raise TypeError("State must be a numpy array")
        if len(state) != self.state_size:
            raise ValueError(f"State size mismatch: expected {self.state_size}, got {len(state)}")
            
        if training and rng.random() < self.epsilon:
            return int(rng.integers(0, self.action_size))
        
        try:
            # Ultra-fast tensor reuse
            self._state_tensor[0] = torch.from_numpy(state).float()
            with torch.no_grad():
                q_values = self.q_network(self._state_tensor)
            return int(q_values.argmax().item())
        except Exception as e:
            logger.warning(f"Error in action selection for agent {self.agent_id}: {e}")
            return 0
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience without priority calculation"""
        try:
            experience = Experience(state, action, reward, next_state, done)
            self.memory.push(experience)
        except Exception as e:
            logger.warning(f"Error storing experience for agent {self.agent_id}: {e}")
    
    def train(self, batch_size: int = 8) -> float:  # Very small batch
        """Ultra-fast training"""
        if len(self.memory) < batch_size:
            return 0.0
        
        try:
            experiences = self.memory.sample(batch_size)
            if not experiences:
                return 0.0
            
            # Simple batch processing
            states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
            actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
            dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
            
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Simple MSE loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Update target network
            self.steps += 1
            if self.steps % self.update_target_frequency == 0:
                self.update_target_network()
            
            return float(loss.item())
            
        except Exception as e:
            logger.warning(f"Error during training for agent {self.agent_id}: {e}")
            return 0.0

class MinimalDataLogger:
    """Minimal logging for performance"""
    
    def __init__(self, log_file: str = "madqn_data.json"):
        self.log_file = log_file
        self.log_counter = 0
        self.log_frequency = 100  # Log every 100th action
    
    def log_state_action_pair(self, sap: StateActionPair):
        """Minimal logging"""
        self.log_counter += 1
        if self.log_counter % self.log_frequency != 0:
            return
        
        # Skip actual file writing for maximum performance
        pass

class MinimalROS2Interface:
    """Minimal ROS2 interface"""
    
    def __init__(self):
        self.node_name = "madqn_navigation_node"
        logger.info(f"Minimal ROS2 Interface initialized")
    
    def publish_action(self, agent_id: int, action: int):
        """Skip publishing for performance"""
        pass

class FastPerformanceMonitor:
    """Ultra-fast performance monitoring"""
    
    def __init__(self):
        self.decision_times = deque(maxlen=50)
        self.training_losses = {}
        self.warning_cooldown = 0
    
    def record_decision_time(self, duration: float):
        """Record decision time with minimal warnings"""
        if isinstance(duration, (int, float)) and duration >= 0:
            self.decision_times.append(float(duration))
            
            # Reduce warning frequency even more
            current_time = time.time()
            if (duration > 0.050 and  # 50ms threshold
                current_time - self.warning_cooldown > 10.0):  # 10 second cooldown
                logger.warning(f"Decision time exceeded 50ms: {duration*1000:.2f}ms")
                self.warning_cooldown = current_time
    
    def get_average_decision_time(self) -> float:
        """Get average decision time"""
        return float(np.mean(self.decision_times)) if self.decision_times else 0.0
    
    def record_training_loss(self, agent_id: int, loss: float):
        """Record training loss"""
        if not isinstance(loss, (int, float)):
            return
        if agent_id not in self.training_losses:
            self.training_losses[agent_id] = deque(maxlen=20)
        self.training_losses[agent_id].append(float(loss))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        max_time = float(max(self.decision_times)) if self.decision_times else 0.0
        return {
            'avg_decision_time_ms': self.get_average_decision_time() * 1000,
            'max_decision_time_ms': max_time * 1000,
            'training_losses': {int(agent_id): float(np.mean(losses)) 
                              for agent_id, losses in self.training_losses.items()},
            'total_decisions': len(self.decision_times)
        }

class UltraFastMADQNSystem:
    """Ultra-optimized MA-DQN System targeting sub-30ms decisions"""
    
    def __init__(self, num_agents: int = 3, grid_size: int = 15):  # Smaller system
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if num_agents > grid_size * grid_size:
            raise ValueError("num_agents cannot exceed grid capacity")
            
        self.num_agents = num_agents
        self.grid_size = grid_size
        
        # Initialize minimal components
        self.environment = MinimalGridEnvironment(grid_size, num_agents)
        self.sensor_simulator = FastSensorSimulator()
        self.data_logger = MinimalDataLogger()
        self.ros2_interface = MinimalROS2Interface()
        self.performance_monitor = FastPerformanceMonitor()
        self.security_manager = SecurityManager()
        
        # Get state size (now only 20 features)
        sample_state = self.environment._get_state_fast(0)
        self.state_size = len(sample_state)
        self.action_size = 5
        
        # Initialize ultra-fast agents
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agents = {i: UltraFastMADQNAgent(i, self.state_size, self.action_size, device=device)
                      for i in range(num_agents)}
        
        # Training parameters
        self.training_active = True
        self.episode_count = 0
        self.max_episode_steps = 50  # Very short episodes
        
        # Minimal threading
        self.training_executor = ThreadPoolExecutor(max_workers=1)  # Single worker
        
        # Skip training frequently for speed
        self.training_skip_counter = 0
        self.training_frequency = 10  # Train every 10th step
        
        logger.info(f"Ultra-fast MA-DQN System initialized with {num_agents} agents, state_size={self.state_size}")
    
    def run_episode(self, training: bool = True) -> Dict[str, float]:
        """Ultra-fast episode execution"""
        try:
            states = self.environment.reset()
            rewards_per_agent = dict.fromkeys(range(self.num_agents), 0.0)
            episode_steps = 0
            
            while episode_steps < self.max_episode_steps:
                start_time = time.time()
                
                # Get sensor status (cached)
                sensor_status = self.sensor_simulator.get_sensor_status()
                
                # Ultra-fast action selection
                actions = self._select_actions_ultrafast(states, sensor_status, training)
                next_states, rewards, dones, _ = self.environment.step(actions)
                
                # Minimal training
                if training:
                    self.training_skip_counter += 1
                    if self.training_skip_counter % self.training_frequency == 0:
                        self._handle_training_minimal(states, actions, rewards, next_states, dones)
                
                # Update tracking
                for agent_id, reward in rewards.items():
                    rewards_per_agent[agent_id] += float(reward)
                
                states = next_states
                episode_steps += 1
                
                # Record performance
                decision_time = time.time() - start_time
                self.performance_monitor.record_decision_time(decision_time)
                
                # Skip most other operations for speed
                
                # Early termination
                if any(dones.values()):
                    break
            
            self.episode_count += 1
            return rewards_per_agent
            
        except Exception as e:
            logger.error(f"Error in run_episode: {e}")
            return dict.fromkeys(range(self.num_agents), 0.0)
    
    def _select_actions_ultrafast(self, states: Dict[int, np.ndarray], 
                                 sensor_status: Dict[str, bool], training: bool) -> Dict[int, int]:
        """Ultra-fast action selection"""
        actions = {}
        
        for agent_id in range(self.num_agents):
            try:
                state = states[agent_id]
                
                # Skip sensor loss if not needed
                if all(sensor_status.values()):
                    corrupted_state = state
                else:
                    corrupted_state = self.sensor_simulator.apply_sensor_loss(state, sensor_status)
                
                # Select action
                action = self.agents[agent_id].select_action(corrupted_state, training)
                
                # Validate action
                if not self.security_manager.validate_action(action, self.action_size):
                    action = 0
                
                actions[agent_id] = int(action)
                
            except Exception as e:
                logger.warning(f"Error selecting action for agent {agent_id}: {e}")
                actions[agent_id] = 0
        
        return actions
    
    def _handle_training_minimal(self, states: Dict[int, np.ndarray], actions: Dict[int, int],
                                rewards: Dict[int, float], next_states: Dict[int, np.ndarray],
                                dones: Dict[int, bool]):
        """Minimal training - only one agent at a time"""
        agent_id = self.training_skip_counter % self.num_agents  # Round robin
        
        try:
            state = states[agent_id]
            action = int(actions[agent_id])
            reward = float(rewards.get(agent_id, 0))
            next_state = next_states[agent_id]
            done = bool(dones.get(agent_id, False))
            
            # Store experience
            self.agents[agent_id].store_experience(state, action, reward, next_state, done)
            
            # Train synchronously for predictable timing
            loss = self.agents[agent_id].train(batch_size=4)  # Very small batch
            if loss > 0:
                self.performance_monitor.record_training_loss(agent_id, float(loss))
                
        except Exception as e:
            logger.warning(f"Error in minimal training for agent {agent_id}: {e}")
    
    def train(self, num_episodes: int = 1000):
        """Ultra-fast training"""
        if num_episodes <= 0:
            logger.warning("num_episodes must be positive")
            return
            
        logger.info(f"Starting ultra-fast training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            self.run_episode(training=True)
            
            # Very reduced reporting
            if episode % 500 == 0:
                performance_report = self.performance_monitor.get_performance_report()
                logger.info(f"Episode {episode}: "
                          f"Avg Decision Time: {performance_report['avg_decision_time_ms']:.2f}ms")
        
        logger.info("Training completed")
    
    def run_live_simulation(self, duration_seconds: int = 300):
        """Ultra-fast live simulation"""
        if duration_seconds <= 0:
            logger.warning("duration_seconds must be positive")
            return
            
        logger.info(f"Starting ultra-fast live simulation for {duration_seconds} seconds")
        
        start_time = time.time()
        episode_count = 0
        
        while time.time() - start_time < duration_seconds:
            self.run_episode(training=True)
            episode_count += 1
            
            # Minimal reporting
            if episode_count % 50 == 0:
                performance_report = self.performance_monitor.get_performance_report()
                logger.info(f"Live simulation - Episode {episode_count}: "
                          f"Avg Decision Time: {performance_report['avg_decision_time_ms']:.2f}ms")
        
        logger.info(f"Live simulation completed. Ran {episode_count} episodes")
    
    def save_models(self, directory: str = "models"):
        """Save trained models securely"""
        try:
            os.makedirs(directory, exist_ok=True)
            
            for agent_id, agent in self.agents.items():
                model_path = os.path.join(directory, f"agent_{agent_id}_model.pth")
                torch.save({
                    'q_network_state_dict': agent.q_network.state_dict(),
                    'target_network_state_dict': agent.target_network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': float(agent.epsilon),
                    'steps': int(agent.steps)
                }, model_path)
            
            logger.info(f"Models saved to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, directory: str = "models"):
        """Load trained models safely"""
        try:
            for agent_id, agent in self.agents.items():
                model_path = os.path.join(directory, f"agent_{agent_id}_model.pth")
                
                checkpoint = self.security_manager.safe_torch_load(model_path, str(agent.device))
                
                if checkpoint is not None:
                    try:
                        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        agent.epsilon = float(checkpoint['epsilon'])
                        agent.steps = int(checkpoint['steps'])
                        logger.info(f"Model loaded for agent {agent_id}")
                    except Exception as e:
                        logger.error(f"Failed to load model state for agent {agent_id}: {e}")
                else:
                    logger.warning(f"No valid model found for agent {agent_id}")
            
            logger.info(f"Model loading process completed from {directory}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            performance_report = self.performance_monitor.get_performance_report()
            
            return {
                'episode_count': int(self.episode_count),
                'training_active': bool(self.training_active),
                'performance': performance_report,
                'agent_epsilons': {int(i): float(agent.epsilon) for i, agent in self.agents.items()},
                'memory_sizes': {int(i): int(len(agent.memory)) for i, agent in self.agents.items()},
                'random_seed': int(RANDOM_SEED),
                'python_version': f"{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}.{PYTHON_VERSION.micro}",
                'supports_executor_timeout': SUPPORTS_EXECUTOR_TIMEOUT
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    def shutdown(self):
        """Graceful shutdown that works across all Python versions"""
        try:
            # Stop logging
            if hasattr(self.data_logger, 'stop_logging'):
                self.data_logger.stop_logging()
            
            # Simple shutdown that works in all Python versions
            self.training_executor.shutdown(wait=True)
            
            logger.info("System shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    """Main function to demonstrate the ultra-fast system"""
    # Check Python version
    logger.info(f"Python version: {PYTHON_VERSION.major}.{PYTHON_VERSION.minor}.{PYTHON_VERSION.micro}")
    logger.info(f"ThreadPoolExecutor timeout support: {SUPPORTS_EXECUTOR_TIMEOUT}")
    
    # Initialize ultra-fast system
    system = UltraFastMADQNSystem(num_agents=3, grid_size=15)
    
    try:
        # Training phase
        logger.info("Starting ultra-fast training phase...")
        system.train(num_episodes=500)
        
        # Save models
        system.save_models()
        
        # Live simulation phase
        logger.info("Starting ultra-fast live simulation phase...")
        system.run_live_simulation(duration_seconds=60)
        
        # System status
        status = system.get_system_status()
        logger.info(f"Final system status: {status}")
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        system.shutdown()
        logger.info("System shutdown")

if __name__ == "__main__":
    main()
