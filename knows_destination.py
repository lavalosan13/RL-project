# import numpy as np
# import gym
# from gym import spaces
# from gym.spaces import Box
# import warnings
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import random

# # Suppress the Box bound precision warning
# warnings.filterwarnings("ignore", category=UserWarning, module="gym.spaces.box")


# # ==============================
# # CLASSE ENVIRONNEMENT (OBSTACLES & DESTINATION)
# # ==============================
# class Environment:
#     """
#     Environnement 3D pour l'apprentissage du drone avec obstacles.
#     """
#     def __init__(self, size=10, p=0.1):
#         self.size = size
#         self.p = p
#         self.obstacles = self.generate_obstacles()
#         self.destination = self.generate_destination()

#     def generate_obstacles(self):
#         obstacles = []
#         for x in range(self.size):
#             for y in range(self.size):
#                 for z in range(self.size):
#                     if np.random.rand() < self.p:
#                         obstacles.append([x, y, z])
#         return np.array(obstacles)

#     def generate_destination(self):
#         return np.random.randint(0, self.size, size=3)
    
#     def render(self):
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_xlim([0, self.size])
#         ax.set_ylim([0, self.size])
#         ax.set_zlim([0, self.size])

#         # Plot obstacles
#         if len(self.obstacles) > 0:
#             ax.scatter(self.obstacles[:, 0], self.obstacles[:, 1], self.obstacles[:, 2], c='r', marker='o', label='Obstacles')

#         # Plot destination
#         ax.scatter(self.destination[0], self.destination[1], self.destination[2], c='g', marker='x', label='Destination')

#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.legend()
#         plt.show()

# # ==============================
# # CLASSE DRONE (DYNAMIQUE & ÉTATS)
# # ==============================
# class Drone:
#     def __init__(self, env):
#         self.position = np.zeros(3, dtype=np.float32)
#         self.velocity = np.zeros(3, dtype=np.float32)
#         self.acceleration = np.zeros(3, dtype=np.float32)
#         self.battery = 100.0
#         self.env = env

#     def reset(self, env):
#         self.position = np.random.uniform(0, env.size, size=3).astype(np.float32)  # Random position within the cube
#         self.velocity = np.zeros(3)
#         self.acceleration = np.zeros(3)
#         self.battery = 100.0
#         self.env = env

#     def generate_random_action(self):
#         """
#         Génère une action aléatoire en suivant une loi uniforme dans chaque direction.
#         """
#         return np.array([random.uniform(-1, 1) for _ in range(3)], dtype=np.float32)

#     def move(self, action):
#         """
#         Met à jour la position du drone selon l'action choisie.
#         Si aucune action n'est fournie, génère une action aléatoire.
#         """
#         if action is None:
#             action = self.generate_random_action()
        
#         action = np.clip(action, -1, 1) * 0.1  # Réduire l'amplitude des actions
#         self.acceleration = action
#         self.velocity += self.acceleration
#         self.position += self.velocity

#         # Consommation d'énergie (proportionnelle à l'accélération)
#         self.battery -= np.sqrt(np.sum(np.abs(self.acceleration)))

#     def check_collision(self):
#         # Check if the drone is within the environment boundaries
#         for coord in self.position:
#             if coord < 0 or coord > self.env.size:
#                 return True
#         # Check if the drone is in an obstacle
#         return any(np.array_equal(self.position, obs) for obs in self.env.obstacles)

#     def calculate_direction(self):
#         """
#         Calculate the direction vector from the drone's current position to the destination.
#         """
#         direction = self.env.destination - self.position
#         direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
#         return direction

#     def detect_obstacles(self, radius=2):
#         """
#         Detect obstacles within a specified radius around the drone.
#         """
#         for obs in self.env.obstacles:
#             if np.linalg.norm(self.position - obs) <= radius:
#                 return True
#         return False

# # ==============================
# # ENVIRONNEMENT GYM (INTERFACE POUR RL)
# # ==============================
# class DroneEnv(gym.Env):
#     def __init__(self):
#         super(DroneEnv, self).__init__()

#         self.env = Environment()
#         self.drone = Drone(self.env)

#         # Définition des espaces Gym
#         self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
#         self.observation_space = spaces.Box(
#             low=np.array([0, 0, 0, -5, -5, -5, 0, 0, 0, 0]),
#             high=np.array([self.env.size, self.env.size, self.env.size, 5, 5, 5, 1000, self.env.size, self.env.size, self.env.size]),
#             dtype=np.float32
#         )

#     def step(self, action=None):
#         if action is None:
#             if not self.drone.detect_obstacles(radius=2):
#                 action = self.drone.calculate_direction()  # Move towards the destination
#             else:
#                 action = self.drone.generate_random_action()  # Generate a random action to avoid obstacles

#         self.drone.move(action)
#         collision = self.drone.check_collision()

#         reward = self.compute_reward(collision)
#         done = collision or self.drone.battery <= 0 or np.array_equal(self.drone.position, self.env.destination)

#         return self._get_obs(), reward, done, {}

#     def compute_reward(self, collision):
#         if np.array_equal(self.drone.position, self.env.destination):
#             return 10000
#         if collision:
#             return -500

#         prev_distance = np.linalg.norm(self.drone.position - self.env.destination)
#         new_distance = np.linalg.norm(self.drone.position + self.drone.velocity - self.env.destination)
#         reward = 100 * (prev_distance - new_distance)

#         # Pénalité énergétique et pour les grandes vitesses
#         reward -= 5 * np.sqrt(np.sum(np.abs(self.drone.acceleration)))
#         reward -= 2 * np.linalg.norm(self.drone.velocity)  # Pénalité pour les grandes vitesses

#         direction_change_penalty = 10 * np.linalg.norm(self.drone.acceleration)
#         reward -= direction_change_penalty
        
#         # Additional penalty for moving towards corners
#         corner_penalty = np.sum(np.abs(self.drone.position - np.array([0, 0, 0]))) + \
#                         np.sum(np.abs(self.drone.position - np.array([self.env.size, self.env.size, self.env.size])))
#         reward -= 0.1 * corner_penalty
#         return reward

#     def reset(self):
#         self.env = Environment()
#         self.drone.reset(self.env)
#         return self._get_obs()

#     def _get_obs(self):
#         return np.concatenate([self.drone.position, self.drone.velocity, [self.drone.battery], self.env.destination])

# # ==============================
# # MODÈLE ACTOR-CRITIC
# # ==============================
# class ActorCritic(nn.Module):
#     """
#     Modèle Actor-Critic pour l'apprentissage par renforcement.

#     Ce modèle utilise deux réseaux de neurones :
#     - Un réseau d'acteur (actor) qui propose des actions basées sur les états actuels.
#     - Un réseau de critique (critic) qui évalue la qualité des actions proposées en estimant la valeur des états.

#     Attributs :
#     ----------
#     fc1 : nn.Linear
#         Première couche linéaire du réseau.
#     fc2 : nn.Linear
#         Deuxième couche linéaire du réseau.
#     actor : nn.Linear
#         Couche linéaire pour le réseau d'acteur, produisant les probabilités d'action.
#     critic : nn.Linear
#         Couche linéaire pour le réseau de critique, produisant la valeur de l'état.

#     Méthodes :
#     ---------
#     forward(x):
#         Passe avant du modèle. Prend un état en entrée et retourne les probabilités d'action et la valeur de l'état.
#     """

#     def __init__(self, input_dim, action_dim):
#         """
#         Initialise le modèle Actor-Critic.

#         Paramètres :
#         -----------
#         input_dim : int
#             Dimension de l'espace d'entrée (nombre de caractéristiques de l'état).
#         action_dim : int
#             Dimension de l'espace d'action (nombre d'actions possibles).
#         """
#         super(ActorCritic, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.actor = nn.Linear(128, action_dim)
#         self.critic = nn.Linear(128, 1)
    
#     def forward(self, x):
#         """
#         Passe avant du modèle.

#         Paramètres :
#         -----------
#         x : torch.Tensor
#             Tensor représentant l'état actuel.

#         Retourne :
#         ---------
#         action_probs : torch.Tensor
#             Tensor représentant les probabilités d'action.
#         state_value : torch.Tensor
#             Tensor représentant la valeur de l'état.
#         """
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         action_means = self.actor(x)
#         state_value = self.critic(x)
#         return action_means, state_value

# # ==============================
# # ENTRAÎNEMENT ACTOR-CRITIC
# # ==============================
# def train(env, model, optimizer, epochs=1, gamma=0.99, reward_threshold=900, window_size=100):
#     """
#     Entraîne le modèle Actor-Critic dans l'environnement spécifié.

#     Paramètres :
#     -----------
#     env : gym.Env
#         L'environnement Gym dans lequel le modèle sera entraîné.
#     model : nn.Module
#         Le modèle Actor-Critic à entraîner.
#     optimizer : torch.optim.Optimizer
#         L'optimiseur utilisé pour mettre à jour les poids du modèle.
#     epochs : int, optionnel (par défaut=1)
#         Le nombre d'épisodes d'entraînement.
#     gamma : float, optionnel (par défaut=0.99)
#         Le facteur de discount pour les récompenses futures.
#     reward_threshold : float, optionnel (par défaut=900)
#         Le seuil de récompense moyenne pour arrêter l'entraînement.
#     window_size : int, optionnel (par défaut=100)
#         Le nombre d'épisodes à considérer pour calculer la récompense moyenne.

#     Retourne :
#     ---------
#     None
#     """
#     reward_history = []
    
#     for episode in range(epochs):
#         state = env.reset()
#         log_probs, values, rewards = [], [], []
#         done = False

#         while not done:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             action_means, state_value = model(state_tensor)
            
#             action = action_means.detach().numpy().flatten()
#             new_state, reward, done, _ = env.step(action)
#             print(f"Action choisie : {action}")
#             log_prob = torch.log(action_means + 1e-10)  # Add small value to avoid log(0)
#             log_probs.append(log_prob)
#             values.append(state_value)
#             rewards.append(reward)
            
#             state = new_state

#         # Calcul des pertes Actor-Critic
#         returns, G = [], 0
#         for r in reversed(rewards):
#             G = r + gamma * G
#             returns.insert(0, G)

#         returns = torch.tensor(returns, dtype=torch.float32)  # Ensure returns are Float tensors
#         values = torch.cat(values).squeeze()
#         advantage = returns - values.detach()
        
#         # Print dimensions of tensors
#         print(f'log_probs dimensions: {[lp.shape for lp in log_probs]}')
#         print(f'values dimensions: {values.shape}')
#         print(f'returns dimensions: {returns.shape}')
#         print(f'advantage dimensions: {advantage.shape}')
#         print(f'logprob is {log_probs}')
#         print(f'advantage is {advantage}')

#         if log_probs:  # Ensure log_probs is not empty
#             log_probs = torch.cat(log_probs).squeeze()  # Ensure log_probs is 1D tensor
#             print(f'log_probs after concatenation: {log_probs.shape}')
#             actor_loss = -(log_probs * advantage.unsqueeze(1)).mean()  # Broadcast advantage to match log_probs
#             critic_loss = F.mse_loss(values, returns)  # Calculer la perte du critique
#             loss = actor_loss + critic_loss  # Calculer la perte totale

#             optimizer.zero_grad()  # Réinitialiser les gradients
#             loss.backward()  # Calculer les gradients
#             optimizer.step()  # Mettre à jour les poids
#         else:
#             print("Warning: log_probs is empty. Skipping loss calculation for this episode.")

#         # Ajouter la récompense totale de l'épisode à l'historique
#         total_reward = sum(rewards)
#         reward_history.append(total_reward)

#         # Vérifier la condition d'arrêt
#         if len(reward_history) >= window_size:
#             avg_reward = np.mean(reward_history[-window_size:])
#             print(f'Episode {episode}, Average Reward: {avg_reward}')
#             if avg_reward >= reward_threshold:
#                 print(f'Condition d\'arrêt atteinte à l\'épisode {episode} avec une récompense moyenne de {avg_reward}')
#                 break

# # ==============================
# # TEST DU MODÈLE ENTRAÎNÉ
# # ==============================
# def test(env, model, max_steps=1000):
#     """
#     Teste le modèle Actor-Critic dans un nouvel environnement et enregistre la trajectoire.

#     Paramètres :
#     -----------
#     env : gym.Env
#         L'environnement Gym dans lequel le modèle sera testé.
#     model : nn.Module
#         Le modèle Actor-Critic à tester.
#     max_steps : int, optionnel (par défaut=1000)
#         Le nombre maximum de pas de temps pour le test.

#     Retourne :
#     ---------
#     total_reward : float
#         La récompense totale obtenue pendant le test.
#     trajectory : list
#         La liste des positions du drone à chaque étape.
#     """
#     state = env.reset()
#     total_reward = 0
#     done = False
#     steps = 0
#     trajectory = []

#     while not done and steps < max_steps:
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
#         action_means, _ = model(state_tensor)
#         action = action_means.detach().numpy().flatten()
#         print(f"Action choisie : {action}")
#         state, reward, done, _ = env.step(action)
#         total_reward += reward
#         steps += 1
#         trajectory.append(env.drone.position.copy())  # Enregistrer la position actuelle du drone

#     return total_reward, trajectory

# def plot_trajectory(env, trajectory):
#     """
#     Visualise la trajectoire du drone dans l'environnement.

#     Paramètres :
#     -----------
#     env : Environment
#         L'environnement contenant les obstacles et la destination.
#     trajectory : list
#         La liste des positions du drone à chaque étape.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim([0, env.size])
#     ax.set_ylim([0, env.size])
#     ax.set_zlim([0, env.size])

#     # Plot obstacles
#     if len(env.obstacles) > 0:
#         ax.scatter(env.obstacles[:, 0], env.obstacles[:, 1], env.obstacles[:, 2], c='r', marker='o', label='Obstacles')

#     # Plot destination
#     ax.scatter(env.destination[0], env.destination[1], env.destination[2], c='g', marker='x', label='Destination')

#     #Plot start point
#     start_point = trajectory[0]
#     ax.scatter(start_point[0], start_point[1], start_point[2], c='b', marker='^', label='Position Initiale du Drone')

#     # Plot trajectory
#     trajectory = np.array(trajectory)
#     ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='b', label='Trajectory')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     plt.show()

# # ==============================
# # EXÉCUTION
# # ==============================
# env = DroneEnv()
# model = ActorCritic(input_dim=10, action_dim=3)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# train(env, model, optimizer, epochs=5000)

# # Test du modèle entraîné
# env_test = DroneEnv()
# total_reward, trajectory = test(env_test, model)
# print(f"Récompense totale obtenue pendant le test : {total_reward}")

# # Visualiser la trajectoire du drone
# plot_trajectory(env_test.env, trajectory)




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

    def render(self, trajectory, reason):
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

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Add reason for stopping
        plt.title(f"Reason for stopping: {reason}")

        plt.show()

# ==============================
# CLASSE DRONE (DYNAMIQUE & ÉTATS)
# ==============================
class Drone:
    def __init__(self, env):
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.battery = 1000.0
        self.env = env

    def reset(self, env):
        while True:
            position = np.random.uniform(0, env.size, size=3).astype(np.float32)  # Random position within the cube
            if not env.is_inside_obstacle(position):
                self.position = position
                break
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.battery = 1000.0
        self.env = env

    def move(self, action):
        """
        Met à jour la position du drone selon l'action choisie.
        """
        if np.array_equal(self.position, self.env.destination):
            return  # Stop moving if the destination is reached

        action = np.clip(action, -1, 1) * 0.03  # Réduire l'amplitude des actions
        self.acceleration = action
        self.velocity += self.acceleration * 0.001
        new_position = self.position + self.velocity

        # Check if the new position is within the boundaries of the cube
        if np.all(new_position >= 0) and np.all(new_position <= self.env.size):
            self.position = new_position

        # Consommation d'énergie (proportionnelle à l'accélération)
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
        Return the positions of detected obstacles.
        """
        detected_obstacles = []
        for obs_center, obs_radius in self.env.obstacles:
            if np.linalg.norm(self.position + direction - obs_center) <= radius + obs_radius:
                detected_obstacles.append(obs_center)
        return detected_obstacles

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
class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.env = Environment()
        self.drone = Drone(self.env)

        # Définition des espaces Gym
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -5, -5, -5, 0, 0, 0, 0]),
            high=np.array([self.env.size, self.env.size, self.env.size, 5, 5, 5, 100, self.env.size, self.env.size, self.env.size]),
            dtype=np.float32
        )

    # def step(self, action=None):
    #     #direction = self.drone.calculate_direction()  # Calculate direction towards the destination
    #     if action is None:
    #         direction = self.drone.calculate_direction()  # Calculate direction towards the destination
    #         if not self.drone.detect_obstacles(direction, radius=0.5):
    #             action = direction  # Move towards the destination
    #         else:
    #             action = self.actor_critic_action()  # Use Actor-Critic model to avoid obstacles

    #     self.drone.move(action)
    #     done = np.array_equal(self.drone.position, self.env.destination) or self.drone.battery <= 0

    #     reward = self.compute_reward(done)
    #     return self._get_obs(), reward, done, {}
    def step(self, action=None):
        direction = self.drone.calculate_direction()  # Calculate direction towards the destination
        if action is None:
            if not self.drone.detect_obstacles(direction, radius=1):
                action = direction  # Move towards the destination
            else:
                action = self.actor_critic_action()  # Use Actor-Critic model to avoid obstacles

        self.drone.move(action)
        done = np.array_equal(self.drone.position, self.env.destination) or self.drone.battery <= 0

        reward = self.compute_reward(done)
        return self._get_obs(), reward, done, {}

    def compute_reward(self, done):
        if done and np.array_equal(self.drone.position, self.env.destination):
            return 100000
        elif done:
            return -50000

        prev_distance = np.linalg.norm(self.drone.position - self.env.destination)
        new_distance = np.linalg.norm(self.drone.position + self.drone.velocity - self.env.destination)
        reward = 10000 * (prev_distance - new_distance)

        # Pénalité énergétique et pour les grandes vitesses
        reward -= 50 * np.sqrt(np.sum(np.abs(self.drone.acceleration)))
        reward -= 20 * np.linalg.norm(self.drone.velocity)  # Pénalité pour les grandes vitesses

        return reward

    def reset(self):
        self.env = Environment()
        self.drone.reset(self.env)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.drone.position, self.drone.velocity, [self.drone.battery], self.env.destination])

    def actor_critic_action(self):
        state = torch.FloatTensor(self._get_obs()).unsqueeze(0)
        action_means, _ = self.model(state)
        action = action_means.detach().numpy().flatten()
        return action

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
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
    
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
        action_means = self.actor(x)
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
            if not env.drone.detect_obstacles(direction, radius=5):
                action = direction  # Move towards the destination
            else:
                action = action_means.detach().numpy().flatten()  # Use Actor-Critic model to avoid obstacles

            #new_state, reward, done, _ = env.step(action)
            new_state, reward, done, _ = env.step()
            print(f"Action choisie : {action}")
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
        print(f'log_probs dimensions: {[lp.shape for lp in log_probs]}')
        print(f'values dimensions: {values.shape}')
        print(f'returns dimensions: {returns.shape}')
        print(f'advantage dimensions: {advantage.shape}')
        print(f'logprob is {log_probs}')
        print(f'advantage is {advantage}')

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
def test(env, model, max_steps=10000):
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
    reason=""

    while not done and steps < max_steps:
    #while not done or steps < max_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_means, _ = model(state_tensor)
        action = action_means.detach().numpy().flatten()
        print(f"Action choisie : {action}")
        #state, reward, done, _ = env.step(action)
        state, reward, done, _ = env.step()
        total_reward += reward
        steps += 1
        trajectory.append(env.drone.position.copy())  # Enregistrer la position actuelle du drone

        if done:
            if np.array_equal(env.drone.position, env.env.destination):
                reason = "Destination atteinte"
            elif env.drone.battery <= 0:
                reason = "Batterie épuisée"
            else:
                reason = "Obstacle rencontré"
    return total_reward, trajectory, reason

# ==============================
# EXÉCUTION
# ==============================
env = DroneEnv()
model = ActorCritic(input_dim=10, action_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
env.model = model  # Assign the model to the environment
train(env, model, optimizer, epochs=10)

# Test du modèle entraîné sur 5 environnements différents
total_rewards = []
for i in range(5):
    env_test = DroneEnv()
    total_reward, trajectory, reason = test(env_test, model)
    total_rewards.append(total_reward)
    print(f"Environnement {i+1} - Récompense totale obtenue pendant le test : {total_reward}")
    # Visualiser la trajectoire du drone
    env_test.env.render(trajectory, reason)

# Afficher la récompense moyenne sur les 5 environnements
average_reward = np.mean(total_rewards)
print(f"Récompense moyenne sur les 5 environnements : {average_reward}")