import numpy as np
import gym
from gym import spaces
from gym.spaces import Box
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Suppress the Box bound precision warning
warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")


# ==============================
# CLASSE ENVIRONNEMENT (OBSTACLES & DESTINATION)
# ==============================
class Environment:
    """
    Environnement 3D pour l'apprentissage du drone avec obstacles.
    """
    def __init__(self, size=10, p=0.1):
        self.size = size
        self.p = p
        self.obstacles = self.generate_obstacles()
        self.destination = self.generate_destination()

    def generate_obstacles(self):
        obstacles = []
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    if np.random.rand() < self.p:
                        obstacles.append([x, y, z])
        return np.array(obstacles)

    def generate_destination(self):
        return np.random.randint(0, self.size, size=3)
    
    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([0, self.size])
        ax.set_ylim([0, self.size])
        ax.set_zlim([0, self.size])

        # Plot obstacles
        if len(self.obstacles) > 0:
            ax.scatter(self.obstacles[:, 0], self.obstacles[:, 1], self.obstacles[:, 2], c='r', marker='o', label='Obstacles')

        # Plot destination
        ax.scatter(self.destination[0], self.destination[1], self.destination[2], c='g', marker='x', label='Destination')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

#env=Environment()
#env.render()
# ==============================
# CLASSE DRONE (DYNAMIQUE & ÉTATS)
# ==============================
class Drone:
    def __init__(self,env):
        self.position = np.zeros(3, dtype=np.float32)  # Initialize position as float64
        self.velocity = np.zeros(3, dtype=np.float32)  # Initialize velocity as float64
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.battery = 100.0
        self.env = env

    def reset(self, env):
        """
        Réinitialise l'état du drone.
        """
        self.position = np.random.uniform(0, self.env.size, size=3)  # Random initial position
        self.velocity = np.zeros(3)  # Vitesse initiale
        self.acceleration = np.zeros(3)  # Accélération initiale
        self.battery = 10000  # Batterie initiale
        self.env = env

    def move(self, action):
        # Ensure action is a 1D array with shape (3,)
        action = np.squeeze(action)  # Remove extra dimensions
        self.acceleration = np.clip(action, -1, 1)  # Clip accelerations between [-1, 1]
        self.velocity += self.acceleration  # Update velocity
        self.position += self.velocity  # Update position

        # Clamp the position to stay within the environment boundaries
        self.position = np.clip(self.position, 0, self.env.size)

        # Energy consumption (proportional to acceleration)
        self.battery -= np.sqrt(np.sum(np.abs(self.acceleration)))

    def check_collision(self):
        #if np.array_equal(self.position, self.env.destination):
            #return False
        #regarder si le drone est dans les limites de l'environnement    
        for coord in self.position:
            if coord <0 or coord>self.env.size :
                return True
        #regarder si le drone est dans un obstacle
        return any(np.array_equal(self.position, obs) for obs in self.env.obstacles)

# ==============================
# ENVIRONNEMENT GYM (INTERFACE POUR RL)
# ==============================
class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.env = Environment()  # Génération du monde
        self.drone = Drone(self.env)  # Initialisation du drone
        self.traj = []
        
        # Définition des espaces Gym
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -5, -5, -5, 0]),  # Position, vitesse, batterie min
            high=np.array([self.env.size, self.env.size, self.env.size, 5, 5, 5, 1000]),  # Max
            dtype=np.float32
        )

    def step(self, action):
        """
        Perform an action and update the drone's state.
        """
        # Ensure action is a NumPy array
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        
        self.drone.move(action)
        self.traj.append(list(self.drone.position.copy()))
        collision = self.drone.check_collision()

        # Calculate reward
        reward = self.compute_reward(collision)

        # Check if the episode is done
        done = collision or self.drone.battery <= 0 or np.array_equal(self.drone.position, self.env.destination)

        return self._get_obs(), reward, done, {}

    def compute_reward(self, collision):
        if np.array_equal(self.drone.position, self.env.destination):
            return 10000  # Large reward for reaching the destination
        if collision:
            return -1000  # Penalty for collision

        # Reward for moving closer to the destination
        distance = np.linalg.norm(self.drone.position - self.env.destination)
        reward = 100 * (1 / (distance + 1))  # Inverse distance reward

        # Penalty for being near obstacles
        for obs in self.env.obstacles:
            obs_distance = np.linalg.norm(self.drone.position - obs)
            if obs_distance < 1.0:  # Penalize if too close to an obstacle
                reward -= 10

        # Small penalty for energy consumption
        reward -= 0.1 * np.sqrt(np.sum(np.abs(self.drone.acceleration)))

        return reward
        
    def plot_trajectory(self):
        """
        Visualise la trajectoire du drone dans l'environnement.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([0, self.env.size])
        ax.set_ylim([0, self.env.size])
        ax.set_zlim([0, self.env.size])

        # Plot obstacles
        if len(self.env.obstacles) > 0:
            ax.scatter(self.env.obstacles[:, 0], self.env.obstacles[:, 1], self.env.obstacles[:, 2], c='r', marker='o', label='Obstacles')

        # Plot destination
        ax.scatter(self.env.destination[0], self.env.destination[1], self.env.destination[2], c='g', marker='x', label='Destination')

        # Plot start point
        if len(self.traj) > 0:
            start_point = self.traj[0]
            ax.scatter(start_point[0], start_point[1], start_point[2], c='b', marker='^', label='Position Initiale du Drone')

        # Plot trajectory
        if len(self.traj) > 0:
            trajectory = np.array(self.traj)
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='b', label='Trajectory')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        self.env = Environment()  # Nouveau monde
        self.drone = Drone(self.env)  # Nouveau drone
        self.traj = []
        return self._get_obs()

    def _get_obs(self):
        """
        Retourne l'état du drone sous forme d'observation Gym.
        """
        return np.concatenate([self.drone.position, self.drone.velocity, [self.drone.battery]])

# ==============================
# MODÈLE ACTOR-CRITIC
# ==============================
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        state_value = self.critic(x)
        return action_mean, action_logstd, state_value
# ==============================
# ENTRAÎNEMENT ACTOR-CRITIC
# ==============================
def train(droneEnv, model, optimizer, epochs=5000, gamma=0.99):
    for episode in range(epochs):
        state = droneEnv.reset()
        log_probs, values, rewards = [], [], []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean, action_logstd, state_value = model(state_tensor)
            
            action_std = torch.exp(action_logstd)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            
            # Add exploration noise
            exploration_noise = torch.randn_like(action) * 0.1
            action = action + exploration_noise
            
            log_prob = dist.log_prob(action).sum(-1)
            
            new_state, reward, done, _ = droneEnv.step(action.detach().numpy())
            
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            
            state = new_state

        # Calculate returns and advantages
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values).squeeze()
        advantage = returns - values.detach()

        if log_probs:
            actor_loss = -(torch.stack(log_probs) * advantage).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            print("Warning: log_probs is empty. Skipping loss calculation for this episode.")

def evaluate(droneEnv, model, episodes=10):
    for episode in range(episodes):
        state = droneEnv.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean, action_logstd, _ = model(state_tensor)
            
            # Sample an action from the Gaussian distribution
            action_std = torch.exp(action_logstd)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            
            # Convert action to a NumPy array and pass it to the environment
            state, reward, done, _ = droneEnv.step(action.detach().numpy())
            total_reward += reward

        # Plot the trajectory after each episode
        droneEnv.plot_trajectory()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# ==============================
# EXÉCUTION
# ==============================
droneEnv = DroneEnv()
model = ActorCritic(input_dim=7, action_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train(droneEnv, model, optimizer, epochs=50)

# Evaluate the trained model
evaluate(droneEnv, model, episodes=6)

