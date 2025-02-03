import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DroneEnv3D:
    def __init__(self, num_obstacles=10):
        self.space_size = 100  # Cube 100x100x100
        self.detection_range = 10  # Portée de détection des obstacles
        
        # Définition des limites d'action et d'observation
        self.action_space_low = np.array([0, 0, 0], dtype=np.float32)  # [vitesse, direction_x, direction_y, direction_z]
        self.action_space_high = np.array([1, 1, 1], dtype=np.float32)
        self.observation_space_low = np.array([0, 0, 0, 0, 0], dtype=np.float32)  # [batterie, x, y, z, distance]
        self.observation_space_high = np.array([100, 100, 100, 100, 100], dtype=np.float32)
        
        # Génération des obstacles et du point de livraison
        self.num_obstacles = num_obstacles
        self.obstacles = self._generate_obstacles()
        self.delivery_point = np.random.uniform(0, self.space_size, size=(3,))
        
        self.reset()
    
    def _generate_obstacles(self):
        """Génère des obstacles fixes aléatoirement dans l'espace 3D"""
        return [np.random.uniform(0, self.space_size, size=(3,)) for _ in range(self.num_obstacles)]
    
    def _detect_obstacles(self):
        """Retourne True si un obstacle est détecté à proximité"""
        for obs in self.obstacles:
            if np.linalg.norm(self.position - obs) <= self.detection_range:
                return True
        return False
    
    def reset(self):
        """Réinitialisation de l'environnement"""
        self.battery = 100  # Batterie pleine
        self.position = np.array([0, 0, 0], dtype=np.float32)  # Position initiale
        self.distance = np.linalg.norm(self.delivery_point - self.position)
        self.path = [self.position.copy()]  # Pour tracer le chemin du drone
        return np.concatenate(([self.battery], self.position, [self.distance]))
    
    def step(self, action):
        """Mise à jour de l'environnement selon l'action du drone"""
        action = np.clip(action, self.action_space_low, self.action_space_high)
        speed, dx, dy, dz = action[0], action[1], action[2], action[3]
        direction = np.array([dx, dy, dz]) / np.linalg.norm([dx, dy, dz])  # Normalisation
        
        # Calcul du déplacement et de la consommation d'énergie
        move_vector = direction * speed * 10  # Déplacement proportionnel à la vitesse
        self.position += move_vector
        self.position = np.clip(self.position, 0, self.space_size)  # Garde dans les limites
        self.battery -= speed * 10  # Consommation d'énergie
        self.distance = np.linalg.norm(self.delivery_point - self.position)
        
        # Tracer le chemin du drone
        self.path.append(self.position.copy())
        
        # Vérification des conditions de fin
        done = self.battery <= 0 or self.distance <= 1  # Atteindre la cible ou batterie vide
        collision = self._detect_obstacles()
        if collision:
            done = True  # Collision met fin à l'épisode
            reward = -100  # Pénalité sévère pour collision
        else:
            reward = -self.distance - (100 - self.battery)  # Éviter une consommation excessive
        
        return np.concatenate(([self.battery], self.position, [self.distance])), reward, done, {}

    def render(self):
        """Affiche l'environnement 3D avec les obstacles et le trajet du drone"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Afficher les obstacles
        for obs in self.obstacles:
            ax.scatter(obs[0], obs[1], obs[2], c='r', marker='o', label="Obstacle")
        
        # Afficher le point de livraison
        ax.scatter(self.delivery_point[0], self.delivery_point[1], self.delivery_point[2], c='g', marker='*', s=100, label="Point de livraison")
        
        # Afficher le chemin du drone
        path = np.array(self.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c='b', label="Trajet du drone")
        
        # Afficher la position du drone
        ax.scatter(self.position[0], self.position[1], self.position[2], c='b', marker='^', s=100, label="Drone")
        
        # Paramètres de l'affichage
        ax.set_xlim(0, self.space_size)
        ax.set_ylim(0, self.space_size)
        ax.set_zlim(0, self.space_size)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        plt.show()

# Exemple d'utilisation
env = DroneEnv3D()
state = env.reset()
print("État initial:", state)

action = [0.5, 1, 0, 0]  # Exemple d'action [vitesse, direction_x, direction_y, direction_z]
state, reward, done, _ = env.step(action)
print("Nouvel état:", state)
print("Récompense:", reward)
print("Terminé:", done)

# Affichage de l'environnement après une étape
env.render()