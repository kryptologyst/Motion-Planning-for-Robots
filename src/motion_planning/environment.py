"""Environment representation for motion planning."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Obstacle:
    """Represents an obstacle in the environment."""
    
    position: np.ndarray
    radius: float = 0.1
    
    def __post_init__(self) -> None:
        """Ensure position is a numpy array."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
    
    def contains(self, point: Union[np.ndarray, Tuple[float, float]]) -> bool:
        """
        Check if a point is inside the obstacle.
        
        Args:
            point: Point to check
            
        Returns:
            True if point is inside obstacle
        """
        point = np.array(point)
        return np.linalg.norm(point - self.position) <= self.radius


class Environment:
    """Represents the planning environment with obstacles and bounds."""
    
    def __init__(self, 
                 bounds: Tuple[float, float, float, float],
                 obstacles: Optional[List[Obstacle]] = None):
        """
        Initialize environment.
        
        Args:
            bounds: Environment bounds as (x_min, y_min, x_max, y_max)
            obstacles: List of obstacles in the environment
        """
        self.bounds = bounds
        self.obstacles = obstacles or []
    
    def is_valid_position(self, position: Union[np.ndarray, Tuple[float, float]]) -> bool:
        """
        Check if a position is valid (within bounds and not in obstacles).
        
        Args:
            position: Position to check
            
        Returns:
            True if position is valid
        """
        position = np.array(position)
        
        # Check bounds
        if not (self.bounds[0] <= position[0] <= self.bounds[2] and
                self.bounds[1] <= position[1] <= self.bounds[3]):
            return False
        
        # Check obstacles
        for obstacle in self.obstacles:
            if obstacle.contains(position):
                return False
        
        return True
    
    def is_valid_path(self, 
                     start: Union[np.ndarray, Tuple[float, float]],
                     end: Union[np.ndarray, Tuple[float, float]],
                     resolution: float = 0.01) -> bool:
        """
        Check if a path between two points is collision-free.
        
        Args:
            start: Start position
            end: End position
            resolution: Resolution for collision checking
            
        Returns:
            True if path is collision-free
        """
        start = np.array(start)
        end = np.array(end)
        
        # Check if both endpoints are valid
        if not (self.is_valid_position(start) and self.is_valid_position(end)):
            return False
        
        # Check intermediate points
        distance = np.linalg.norm(end - start)
        num_steps = int(np.ceil(distance / resolution))
        
        for i in range(num_steps + 1):
            t = i / num_steps if num_steps > 0 else 0
            point = start + t * (end - start)
            if not self.is_valid_position(point):
                return False
        
        return True
    
    def add_obstacle(self, position: Union[np.ndarray, Tuple[float, float]], 
                    radius: float = 0.1) -> None:
        """
        Add an obstacle to the environment.
        
        Args:
            position: Obstacle position
            radius: Obstacle radius
        """
        obstacle = Obstacle(position=np.array(position), radius=radius)
        self.obstacles.append(obstacle)
    
    def get_random_position(self) -> np.ndarray:
        """
        Get a random valid position in the environment.
        
        Returns:
            Random valid position
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(self.bounds[0], self.bounds[2])
            y = np.random.uniform(self.bounds[1], self.bounds[3])
            position = np.array([x, y])
            
            if self.is_valid_position(position):
                return position
        
        # Fallback: return center if no valid position found
        center_x = (self.bounds[0] + self.bounds[2]) / 2
        center_y = (self.bounds[1] + self.bounds[3]) / 2
        return np.array([center_x, center_y])
