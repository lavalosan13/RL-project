#######################################################
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==============================
# CLASSE ENVIRONNEMENT (DESTINATION ET OBSTACLES)
# ==============================

center = (5,5,5)
radius = 2
Obstacle = [(center, radius)]
Destination = (9,5,5)

class EnvironmentManuel:
    """
    Environnement 3D pour l'apprentissage du drone avec obstacles.
    """
    def __init__(self, size=10, obstacle=Obstacle, destination=Destination):
        self.size = size
        self.obstacles = obstacle  # Initialize obstacles first
        self.destination = destination  # Then generate destination

    def is_inside_obstacle(self, position):
        for center, radius in self.obstacles:
            if np.linalg.norm(position - center) <= radius:
                return True
        return False

    def render(self, trajectory, reason, eps=0.5):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([0, self.size])
        ax.set_ylim([0, self.size])
        ax.set_zlim([0, self.size])

        # Plot destination
        ax.scatter(self.destination[0], self.destination[1], self.destination[2], c='g', marker='x', label='Destination')

        # Plot start point
        start_point = trajectory[0]
        ax.scatter(start_point[0], start_point[1], start_point[2], c='b', marker='^', label='Position Initiale du Drone')

        # Plot trajectory
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='b', label='Trajectory')

        # Plot obstacles
        for center, radius in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color="r", alpha=0.5)

        # # Add green triangle if the drone reaches its destination
        # if reason == "Destination atteinte":
        #     # Define the vertices of the triangle
        #     vertices = np.array([
        #         [trajectory[-1][0], trajectory[-1][1], trajectory[-1][2]],
        #         [trajectory[-1][0] + eps, trajectory[-1][1], trajectory[-1][2]],
        #         [trajectory[-1][0], trajectory[-1][1] + eps, trajectory[-1][2]]
        #     ])
        #     # Define the faces of the triangle
        #     faces = [[0, 1, 2]]
        #     # Plot the triangle
        #     ax.add_collection3d(Poly3DCollection([vertices], color='g', alpha=0.5, label='Zone de détection'))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Add reason for stopping
        plt.title(f"Reason for stopping: {reason}")

        plt.show()


class Environment:
    """
    Environnement 3D pour l'apprentissage du drone avec obstacles.
    """
    def __init__(self, size=10):
        self.size = size
        self.obstacles = self.generate_obstacles()  # Initialize obstacles first
        self.destination = self.generate_destination()  # Then generate destination

    def generate_destination(self):
        while True:
            destination = np.random.randint(0, self.size, size=3)
            if not self.is_inside_obstacle(destination):
                return destination

    def generate_obstacles(self):
        obstacles = []
        num_obstacles = np.random.randint(0, 10)
        for _ in range(num_obstacles):
            # Random position for the center of the sphere
            center = np.random.uniform(0, self.size, size=3)
            # Random radius for the sphere
            radius = np.random.uniform(0.5, self.size / 4)
            obstacles.append((center, radius))
        return obstacles

    def is_inside_obstacle(self, position):
        for center, radius in self.obstacles:
            if np.linalg.norm(position - center) <= radius:
                return True
        return False

    def render(self, trajectory, reason, eps=0.5):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([0, self.size])
        ax.set_ylim([0, self.size])
        ax.set_zlim([0, self.size])

        # Plot destination
        ax.scatter(self.destination[0], self.destination[1], self.destination[2], c='g', marker='x', label='Destination')

        # Plot start point
        start_point = trajectory[0]
        ax.scatter(start_point[0], start_point[1], start_point[2], c='b', marker='^', label='Position Initiale du Drone')

        # Plot trajectory
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='b', label='Trajectory')

        # Plot obstacles
        for center, radius in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color="r", alpha=0.5)

        # # Add green triangle if the drone reaches its destination
        # if reason == "Destination atteinte":
        #     # Define the vertices of the triangle
        #     vertices = np.array([
        #         [trajectory[-1][0], trajectory[-1][1], trajectory[-1][2]],
        #         [trajectory[-1][0] + eps, trajectory[-1][1], trajectory[-1][2]],
        #         [trajectory[-1][0], trajectory[-1][1] + eps, trajectory[-1][2]]
        #     ])
        #     # Define the faces of the triangle
        #     faces = [[0, 1, 2]]
        #     # Plot the triangle
        #     ax.add_collection3d(Poly3DCollection([vertices], color='g', alpha=0.5, label='Zone de détection'))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Add reason for stopping
        plt.title(f"Reason for stopping: {reason}")

        plt.show()

Environnement = EnvironmentManuel()

# ==============================
# CLASSE DRONE (DYNAMIQUE & ÉTATS)
# ==============================
class Drone:
    def __init__(self, Environnement, eps=0.5):  # Increased from 0.5 to 1.0
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.battery = 10000.0
        self.env = Environnement
        self.eps = eps  # Larger detection radius

    def reset(self, env):
        while True:
            position = np.random.uniform(0, env.size, size=3).astype(np.float32)  # Random position within the cube
            if not env.is_inside_obstacle(position):
                self.position = position
                break
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.battery = 10000.0

    def move(self, action):
        if np.linalg.norm(self.position - self.env.destination) <= self.eps:
            return  # Stop if destination is reached
        
        action = np.clip(action, -1, 1)
        self.acceleration = action * 0.01  # Smaller acceleration for finer control
        
        # Reduce velocity near the destination
        distance = np.linalg.norm(self.position - self.env.destination)
        if distance < 5.0:
            self.velocity *= 0.5  # Dampen velocity near the goal
        
        self.velocity = np.clip(self.velocity + self.acceleration * 0.1, -2.0, 2.0)  # Clamp velocity
        new_position = self.position + self.velocity
        
        # Boundary check
        if np.all(new_position >= 0) and np.all(new_position <= self.env.size):
            self.position = new_position
        
        self.battery -= np.sqrt(np.sum(np.abs(self.acceleration)))

    def calculate_direction(self):
        """
        Calculate the direction vector from the drone's current position to the destination.
        """
        direction = self.env.destination - self.position
        direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
        return direction
    
    def detect_obstacles(self, direction, radius=10):
        """
        Detect obstacles within a specified radius along the direction vector.
        Return True if an obstacle is detected, otherwise False.
        """
        for obs_center, obs_radius in self.env.obstacles:
            if np.linalg.norm(self.position + direction - obs_center) <= radius + obs_radius:
                return True
        return False

    def is_obstacle_in_traj(self, trajectory):
        """
        Check if the trajectory passes through any detected obstacles.
        """
        for point in trajectory:
            if self.detect_obstacles_at_point(point):
                return True
        return False

    def detect_obstacles_at_point(self, point, radius=10):
        """
        Detect obstacles at a specific point within a specified radius.
        """
        for obs_center, obs_radius in self.env.obstacles:
            if np.linalg.norm(point - obs_center) <= radius + obs_radius:
                return True
        return False

# ==============================
# ENVIRONNEMENT GYM (INTERFACE POUR RL)
# ==============================

class DroneEnv():
    def __init__(self,Environnement):
        super(DroneEnv, self).__init__()

        self.env = Environnement
        self.drone = Drone(self.env)

        # Définition des espaces Gym
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -5, -5, -5, 0, 0, 0, 0]),
            high=np.array([self.env.size, self.env.size, self.env.size, 5, 5, 5, 100, self.env.size, self.env.size, self.env.size]),
            dtype=np.float32
        )

    def check_collision(self):
        #if np.array_equal(self.position, self.env.destination):
            #return False
        #regarder si le drone est dans un obstacle
        hit_obstacle = any(np.linalg.norm(self.drone.position - obs_center) <= self.drone.eps+obs_radium for obs_center, obs_radium in self.env.obstacles)
        print(hit_obstacle)
        #regarder si le drone est dans les limites de l'environnement   
        hit_boundary = np.any(self.drone.position < 0) or np.any(self.drone.position > self.env.size)
        print(hit_boundary)
        collison = any([hit_obstacle, hit_boundary])
        return collison


    def step(self, action=None):
        direction = self.drone.calculate_direction()  # Calculate direction towards the destination
        if action is None:
            if not self.drone.detect_obstacles(direction, radius=1):
                action = direction  # Move towards the destination
            else:
                action = self.actor_critic_action()  # Use Actor-Critic model to avoid obstacles

        self.drone.move(action)
        reached_destination = np.linalg.norm(self.drone.position - self.env.destination) <= self.drone.eps
        battery_depleted = self.drone.battery <= 0
        collison = self.check_collision()
        done = reached_destination or collison or battery_depleted
        reward = self.compute_reward(reached_destination, collison, battery_depleted)
        return self._get_obs(), reward, done, {}

    def compute_reward(self, reached_destination, collision, battery_depleted):
        if reached_destination:
            return 100000  # Success reward
        if collision or battery_depleted:
            return -50000  # Terminal penalty
        
        # Distance-based shaping reward
        distance = np.linalg.norm(self.drone.position - self.env.destination)
        proximity_reward = 1000 * (1 / (distance + 1))  # Prioritize proximity
        
        # Obstacle proximity penalty (prevent circling)
        obstacle_penalty = 0
        for center, radius in self.env.obstacles:
            obs_distance = np.linalg.norm(self.drone.position - center)
            if obs_distance < radius + 2.0:  # Penalize proximity to obstacles
                obstacle_penalty -= 500 * (1 / (obs_distance + 1e-6))  # Avoid division by zero
        
        # Reduce velocity/acceleration penalties near the destination
        penalty_scale = max(0.1, distance / self.env.size)  # Scale penalties with distance
        acceleration_penalty = 10 * penalty_scale * np.sqrt(np.sum(np.abs(self.drone.acceleration)))
        velocity_penalty = 5 * penalty_scale * np.linalg.norm(self.drone.velocity)
        
        total_reward = proximity_reward + obstacle_penalty - acceleration_penalty - velocity_penalty
        return total_reward

    def reset(self):
        self.drone.reset(self.env)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.drone.position, self.drone.velocity, [self.drone.battery], self.env.destination])

    def actor_critic_action(self):
        state = torch.FloatTensor(self._get_obs()).unsqueeze(0)
        action_means, _ = self.model(state)
        action = action_means.detach().numpy().flatten()
        return action

    def can_reach_destination(self):
        """
        Check if the drone can reach the destination in a straight line without hitting any obstacles.
        """
        direction = self.drone.calculate_direction()
        for obs_center, obs_radius in self.env.obstacles:
            if np.linalg.norm(self.drone.position + direction - obs_center) <= obs_radius:
                return False
        return True

# ==============================
# MODÈLE ACTOR-CRITIC
# ==============================
class ActorCritic(nn.Module):
    """
    Modèle Actor-Critic pour l'apprentissage par renforcement.

    Ce modèle utilise deux réseaux de neurones :
    - Un réseau d'acteur (actor) qui propose des actions basées sur les états actuels.
    - Un réseau de critique (critic) qui évalue la qualité des actions proposées en estimant la valeur des états.

    Attributs :
    ----------
    fc1 : nn.Linear
        Première couche linéaire du réseau.
    fc2 : nn.Linear
        Deuxième couche linéaire du réseau.
    actor : nn.Linear
        Couche linéaire pour le réseau d'acteur, produisant les probabilités d'action.
    critic : nn.Linear
        Couche linéaire pour le réseau de critique, produisant la valeur de l'état.

    Méthodes :
    ---------
    forward(x):
        Passe avant du modèle. Prend un état en entrée et retourne les probabilités d'action et la valeur de l'état.
    """

    def __init__(self, input_dim, action_dim):
        """
        Initialise le modèle Actor-Critic.

        Paramètres :
        -----------
        input_dim : int
            Dimension de l'espace d'entrée (nombre de caractéristiques de l'état).
        action_dim : int
            Dimension de l'espace d'action (nombre d'actions possibles).
        """
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Increased layer size
        self.fc2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Passe avant du modèle.

        Paramètres :
        -----------
        x : torch.Tensor
            Tensor représentant l'état actuel.

        Retourne :
        ---------
        action_probs : torch.Tensor
            Tensor représentant les probabilités d'action.
        state_value : torch.Tensor
            Tensor représentant la valeur de l'état.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_means = torch.tanh(self.actor(x))  # Tanh to keep actions in [-1, 1]
        state_value = self.critic(x)
        return action_means, state_value

# ==============================
# ENTRAÎNEMENT ACTOR-CRITIC
# ==============================
def train(env, model, optimizer, epochs=1, gamma=0.99, reward_threshold=900, window_size=100):
    """
    Entraîne le modèle Actor-Critic dans l'environnement spécifié.

    Paramètres :
    -----------
    env : gym.Env
        L'environnement Gym dans lequel le modèle sera entraîné.
    model : nn.Module
        Le modèle Actor-Critic à entraîner.
    optimizer : torch.optim.Optimizer
        L'optimiseur utilisé pour mettre à jour les poids du modèle.
    epochs : int, optionnel (par défaut=1)
        Le nombre d'épisodes d'entraînement.
    gamma : float, optionnel (par défaut=0.99)
        Le facteur de discount pour les récompenses futures.
    reward_threshold : float, optionnel (par défaut=900)
        Le seuil de récompense moyenne pour arrêter l'entraînement.
    window_size : int, optionnel (par défaut=100)
        Le nombre d'épisodes à considérer pour calculer la récompense moyenne.

    Retourne :
    ---------
    None
    """
    reward_history = []
    
    for episode in range(epochs):
        state = env.reset()
        log_probs, values, rewards = [], [], []
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_means, state_value = model(state_tensor)
            
            # Calculate direction and check for obstacles
            direction = env.drone.calculate_direction()
            if not env.drone.detect_obstacles(direction, radius=1):
                action = direction  # Move towards the destination
            else:
                action = action_means.detach().numpy().flatten()  # Use Actor-Critic model to avoid obstacles

            new_state, reward, done, _ = env.step(action)
            #print(f"Action choisie : {action}")
            log_prob = torch.log(action_means + 1e-10)  # Add small value to avoid log(0)
            log_probs.append(log_prob)
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
        
        # Print dimensions of tensors
        #print(f'log_probs dimensions: {[lp.shape for lp in log_probs]}')
        #print(f'values dimensions: {values.shape}')
        #print(f'returns dimensions: {returns.shape}')
        #print(f'advantage dimensions: {advantage.shape}')
        #print(f'logprob is {log_probs}')
        #print(f'advantage is {advantage}')

        if log_probs:  # Ensure log_probs is not empty
            log_probs = torch.cat(log_probs).squeeze()  # Ensure log_probs is 1D tensor
            print(f'log_probs after concatenation: {log_probs.shape}')
            actor_loss = -(log_probs * advantage.unsqueeze(1)).mean()  # Broadcast advantage to match log_probs
            critic_loss = F.mse_loss(values, returns)  # Calculer la perte du critique
            loss = actor_loss + critic_loss  # Calculer la perte totale

            optimizer.zero_grad()  # Réinitialiser les gradients
            loss.backward()  # Calculer les gradients
            optimizer.step()  # Mettre à jour les poids
        else:
            print("Warning: log_probs is empty. Skipping loss calculation for this episode.")

        # Ajouter la récompense totale de l'épisode à l'historique
        total_reward = sum(rewards)
        reward_history.append(total_reward)

        # Vérifier la condition d'arrêt
        if len(reward_history) >= window_size:
            avg_reward = np.mean(reward_history[-window_size:])
            print(f'Episode {episode}, Average Reward: {avg_reward}')
            if avg_reward >= reward_threshold:
                print(f'Condition d\'arrêt atteinte à l\'épisode {episode} avec une récompense moyenne de {avg_reward}')
                break

# ==============================
# TEST DU MODÈLE ENTRAÎNÉ
# ==============================
def test(env, model, max_steps=100000):
    """
    Teste le modèle Actor-Critic dans un nouvel environnement et enregistre la trajectoire.

    Paramètres :
    -----------
    env : gym.Env
        L'environnement Gym dans lequel le modèle sera testé.
    model : nn.Module
        Le modèle Actor-Critic à tester.
    max_steps : int, optionnel (par défaut=1000)
        Le nombre maximum de pas de temps pour le test.

    Retourne :
    ---------
    total_reward : float
        La récompense totale obtenue pendant le test.
    trajectory : list
        La liste des positions du drone à chaque étape.
    """
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    trajectory = []
    reason = ""

    while not done and steps < max_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_means, _ = model(state_tensor)
        action = action_means.detach().numpy().flatten()
        print(f"Action choisie : {action}")
#        state, reward, done, _ = env.step(action)
        state, reward, done, _ = env.step()
        total_reward += reward
        steps += 1
        trajectory.append(env.drone.position.copy())  # Enregistrer la position actuelle du drone

        if done:
            if np.linalg.norm(env.drone.position - env.env.destination) <= env.drone.eps:
                reason = "Destination atteinte"
            elif env.drone.battery <= 0:
                reason = "Batterie épuisée"
            else:
                reason = "Obstacle rencontré"
        else:
            reason = "Sortie de l'espace d'étude"
    
    return total_reward, trajectory, reason

# ==============================
# EXÉCUTION
# ==============================
env = DroneEnv(Environnement)
model = ActorCritic(input_dim=10, action_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.0003)  # Lower learning rate
env.model = model  # Assign the model to the environment
train(env, model, optimizer, epochs=50)

# Test du modèle entraîné sur 5 environnements différents
total_rewards = []
for i in range(10):
    env_test = DroneEnv(Environnement)
    env_test.model = model  # Assign the model to the test environment
    total_reward, trajectory, reason = test(env_test, model)
    total_rewards.append(total_reward)
    print(f"Récompense totale obtenue pendant le test {i+1} : {total_reward}")
    # Visualiser la trajectoire du drone
    env_test.env.render(trajectory, reason, eps=env_test.drone.eps)

# Afficher la récompense moyenne sur les 5 environnements
average_reward = np.mean(total_rewards)
print(f"Récompense moyenne sur les 5 tests : {average_reward}")