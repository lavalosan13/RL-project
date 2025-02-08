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
    def __init__(self, size=10, p=0.001):
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

    def reset(self,env):
        """
        Réinitialise l'état du drone.
        """
        self.position = np.array([10, 10, 10])  # Position de départ
        self.velocity = np.zeros(3)  # Vitesse initiale
        self.acceleration = np.zeros(3)  # Accélération initiale
        self.battery = 1000  # Batterie initiale
        self.env = env

    def move(self, action):
        """
        Met à jour la position du drone selon l'action choisie.
        """
        self.acceleration = np.clip(action, -1, 1)  # Clip les accélérations entre [-1,1]
        self.velocity += self.acceleration  # Mise à jour de la vitesse
        self.position += self.velocity  # Mise à jour de la position

        # Consommation d'énergie (proportionnelle à l'accélération)
        self.battery -= np.sqrt(np.sum(np.abs(self.acceleration)))

    def check_collision(self):
        """
        Vérifie si le drone a percuté un obstacle.
        """
        return any(np.array_equal(self.position, obs) for obs in self.env.obstacles)

# ==============================
# ENVIRONNEMENT GYM (INTERFACE POUR RL)
# ==============================
class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.env = Environment()  # Génération du monde
        self.drone = Drone(self.env)  # Initialisation du drone
        
        # Définition des espaces Gym
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -5, -5, -5, 0]),  # Position, vitesse, batterie min
            high=np.array([self.env.size, self.env.size, self.env.size, 5, 5, 5, 1000]),  # Max
            dtype=np.float32
        )

    def step(self, action):
        """
        Effectue une action et met à jour l'état du drone.
        """
        self.drone.move(action)
        collision = self.drone.check_collision()

        # Calcul de la récompense
        reward = self.compute_reward(collision)

        # Vérification de fin d'épisode
        done = collision or self.drone.battery <= 0 or np.array_equal(self.drone.position, self.env.destination)

        return self._get_obs(), reward, done, {}

    def compute_reward(self, collision):
        """
        Fonction de récompense pour l'agent RL.
        """
        if np.array_equal(self.drone.position, self.env.destination):
            return 1000  # Récompense élevée si la livraison est réussie
        if collision:
            return -500  # Pénalité forte en cas de collision

        # Encouragement à se rapprocher de la destination
        prev_distance = np.linalg.norm(self.drone.position - self.env.destination)
        new_distance = np.linalg.norm(self.drone.position + self.drone.velocity - self.env.destination)
        reward = 10 * (prev_distance - new_distance)

        # Pénalité énergétique
        reward -= 5 * np.sqrt(np.sum(np.abs(self.drone.acceleration)))

        return reward

    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        self.env = Environment()  # Nouveau monde
        self.drone = Drone(self.env)  # Nouveau drone
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
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# ==============================
# ENTRAÎNEMENT ACTOR-CRITIC
# ==============================
def train(env, model, optimizer, epochs=5000, gamma=0.99):
    for episode in range(epochs):
        state = env.reset()
        log_probs, values, rewards = [], [], []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = model(state_tensor)
            
            action = torch.multinomial(action_probs, 1).item()
            new_state, reward, done, _ = env.step(action)
            print(action_probs)
            log_probs.append(torch.log(action_probs[0, action]).unsqueeze(0))  # Ensure log_probs are 1D tensors
            values.append(state_value)
            rewards.append(reward)
            
            state = new_state

        # Calcul des pertes Actor-Critic
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)  # Ensure returns are Float tensors
        values = torch.cat(values).squeeze()
        advantage = returns - values.detach()
        print(f'logprob is {log_probs}')
        print(f'advantage is {advantage}')

        if log_probs:  # Ensure log_probs is not empty
            actor_loss = -(torch.cat(log_probs) * advantage).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            print("Warning: log_probs is empty. Skipping loss calculation for this episode.")
# ==============================
# EXÉCUTION
# ==============================
env = DroneEnv()
model = ActorCritic(input_dim=7, action_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(env, model, optimizer, epochs=1)

print(model)