"""Motion planning algorithms implementation."""

from __future__ import annotations

import numpy as np
import random
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from .environment import Environment
from .utils import Node, Path, distance


class MotionPlanner(ABC):
    """Abstract base class for motion planners."""
    
    def __init__(self, environment: Environment, **kwargs):
        """
        Initialize motion planner.
        
        Args:
            environment: Planning environment
            **kwargs: Additional planner-specific parameters
        """
        self.environment = environment
    
    @abstractmethod
    def plan(self, start: Union[np.ndarray, Tuple[float, float]], 
             goal: Union[np.ndarray, Tuple[float, float]]) -> Optional[Path]:
        """
        Plan a path from start to goal.
        
        Args:
            start: Start position
            goal: Goal position
            
        Returns:
            Planned path or None if no path found
        """
        pass


class RRT(MotionPlanner):
    """Rapidly-exploring Random Tree (RRT) planner."""
    
    def __init__(self, 
                 environment: Environment,
                 max_iterations: int = 1000,
                 step_size: float = 0.1,
                 goal_tolerance: float = 0.1,
                 random_seed: Optional[int] = None):
        """
        Initialize RRT planner.
        
        Args:
            environment: Planning environment
            max_iterations: Maximum number of iterations
            step_size: Step size for tree expansion
            goal_tolerance: Tolerance for reaching goal
            random_seed: Random seed for reproducibility
        """
        super().__init__(environment)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def plan(self, start: Union[np.ndarray, Tuple[float, float]], 
             goal: Union[np.ndarray, Tuple[float, float]]) -> Optional[Path]:
        """
        Plan a path using RRT algorithm.
        
        Args:
            start: Start position
            goal: Goal position
            
        Returns:
            Planned path or None if no path found
        """
        start = np.array(start)
        goal = np.array(goal)
        
        # Validate start and goal positions
        if not self.environment.is_valid_position(start):
            raise ValueError("Start position is not valid")
        if not self.environment.is_valid_position(goal):
            raise ValueError("Goal position is not valid")
        
        # Initialize tree with start node
        start_node = Node(position=start)
        tree = [start_node]
        
        for _ in range(self.max_iterations):
            # Generate random node
            if random.random() < 0.1:  # 10% chance to sample goal
                random_pos = goal
            else:
                random_pos = self.environment.get_random_position()
            
            # Find nearest node in tree
            nearest_node = min(tree, key=lambda n: distance(n.position, random_pos))
            
            # Step towards random position
            direction = random_pos - nearest_node.position
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
                new_pos = nearest_node.position + self.step_size * direction
                
                # Check if path to new position is collision-free
                if self.environment.is_valid_path(nearest_node.position, new_pos):
                    new_node = Node(position=new_pos, parent=nearest_node)
                    tree.append(new_node)
                    
                    # Check if goal is reached
                    if distance(new_pos, goal) <= self.goal_tolerance:
                        goal_node = Node(position=goal, parent=new_node)
                        tree.append(goal_node)
                        return self._reconstruct_path(goal_node)
        
        return None
    
    def _reconstruct_path(self, goal_node: Node) -> Path:
        """Reconstruct path from goal to start."""
        path_nodes = []
        current = goal_node
        
        while current is not None:
            path_nodes.append(current)
            current = current.parent
        
        path_nodes.reverse()
        return Path(nodes=path_nodes)


class RRTStar(RRT):
    """RRT* planner with optimal path refinement."""
    
    def __init__(self, 
                 environment: Environment,
                 max_iterations: int = 1000,
                 step_size: float = 0.1,
                 goal_tolerance: float = 0.1,
                 rewire_radius: float = 0.5,
                 random_seed: Optional[int] = None):
        """
        Initialize RRT* planner.
        
        Args:
            environment: Planning environment
            max_iterations: Maximum number of iterations
            step_size: Step size for tree expansion
            goal_tolerance: Tolerance for reaching goal
            rewire_radius: Radius for rewiring connections
            random_seed: Random seed for reproducibility
        """
        super().__init__(environment, max_iterations, step_size, goal_tolerance, random_seed)
        self.rewire_radius = rewire_radius
    
    def plan(self, start: Union[np.ndarray, Tuple[float, float]], 
             goal: Union[np.ndarray, Tuple[float, float]]) -> Optional[Path]:
        """
        Plan a path using RRT* algorithm.
        
        Args:
            start: Start position
            goal: Goal position
            
        Returns:
            Planned path or None if no path found
        """
        start = np.array(start)
        goal = np.array(goal)
        
        # Validate start and goal positions
        if not self.environment.is_valid_position(start):
            raise ValueError("Start position is not valid")
        if not self.environment.is_valid_position(goal):
            raise ValueError("Goal position is not valid")
        
        # Initialize tree with start node
        start_node = Node(position=start, cost=0.0)
        tree = [start_node]
        goal_node = None
        
        for _ in range(self.max_iterations):
            # Generate random node
            if random.random() < 0.1:  # 10% chance to sample goal
                random_pos = goal
            else:
                random_pos = self.environment.get_random_position()
            
            # Find nearest node in tree
            nearest_node = min(tree, key=lambda n: distance(n.position, random_pos))
            
            # Step towards random position
            direction = random_pos - nearest_node.position
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
                new_pos = nearest_node.position + self.step_size * direction
                
                # Check if path to new position is collision-free
                if self.environment.is_valid_path(nearest_node.position, new_pos):
                    # Find nodes within rewire radius
                    nearby_nodes = [
                        node for node in tree 
                        if distance(node.position, new_pos) <= self.rewire_radius
                    ]
                    
                    # Find best parent (lowest cost)
                    best_parent = nearest_node
                    best_cost = nearest_node.cost + distance(nearest_node.position, new_pos)
                    
                    for node in nearby_nodes:
                        if self.environment.is_valid_path(node.position, new_pos):
                            cost = node.cost + distance(node.position, new_pos)
                            if cost < best_cost:
                                best_parent = node
                                best_cost = cost
                    
                    # Create new node
                    new_node = Node(position=new_pos, parent=best_parent, cost=best_cost)
                    tree.append(new_node)
                    
                    # Rewire nearby nodes
                    for node in nearby_nodes:
                        if (not np.array_equal(node.position, best_parent.position) and 
                            self.environment.is_valid_path(new_pos, node.position)):
                            new_cost = new_node.cost + distance(new_pos, node.position)
                            if new_cost < node.cost:
                                node.parent = new_node
                                node.cost = new_cost
                    
                    # Check if goal is reached
                    if distance(new_pos, goal) <= self.goal_tolerance:
                        goal_node = Node(position=goal, parent=new_node, 
                                       cost=new_node.cost + distance(new_pos, goal))
                        tree.append(goal_node)
        
        if goal_node is not None:
            return self._reconstruct_path(goal_node)
        
        return None


class AStar(MotionPlanner):
    """A* planner for grid-based environments."""
    
    def __init__(self, 
                 environment: Environment,
                 resolution: float = 0.1,
                 heuristic_weight: float = 1.0):
        """
        Initialize A* planner.
        
        Args:
            environment: Planning environment
            resolution: Grid resolution
            heuristic_weight: Weight for heuristic function
        """
        super().__init__(environment)
        self.resolution = resolution
        self.heuristic_weight = heuristic_weight
    
    def plan(self, start: Union[np.ndarray, Tuple[float, float]], 
             goal: Union[np.ndarray, Tuple[float, float]]) -> Optional[Path]:
        """
        Plan a path using A* algorithm.
        
        Args:
            start: Start position
            goal: Goal position
            
        Returns:
            Planned path or None if no path found
        """
        start = np.array(start)
        goal = np.array(goal)
        
        # Validate start and goal positions
        if not self.environment.is_valid_position(start):
            raise ValueError("Start position is not valid")
        if not self.environment.is_valid_position(goal):
            raise ValueError("Goal position is not valid")
        
        # Create grid
        x_min, y_min, x_max, y_max = self.environment.bounds
        width = int((x_max - x_min) / self.resolution)
        height = int((y_max - y_min) / self.resolution)
        
        # Convert positions to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)
        
        # A* search
        open_set = {start_grid}
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        while open_set:
            current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
            
            if current == goal_grid:
                return self._reconstruct_path(came_from, current, start_grid)
            
            open_set.remove(current)
            
            # Check neighbors
            for neighbor in self._get_neighbors(current, width, height):
                if not self.environment.is_valid_position(self._grid_to_world(neighbor)):
                    continue
                
                tentative_g_score = g_score[current] + distance(
                    self._grid_to_world(current), 
                    self._grid_to_world(neighbor)
                )
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic_weight * self._heuristic(neighbor, goal_grid)
                    
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        
        return None
    
    def _world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        x_min, y_min, _, _ = self.environment.bounds
        x = int((pos[0] - x_min) / self.resolution)
        y = int((pos[1] - y_min) / self.resolution)
        return (x, y)
    
    def _grid_to_world(self, pos: Tuple[int, int]) -> np.ndarray:
        """Convert grid coordinates to world coordinates."""
        x_min, y_min, _, _ = self.environment.bounds
        x = x_min + pos[0] * self.resolution
        y = y_min + pos[1] * self.resolution
        return np.array([x, y])
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance between grid positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _get_neighbors(self, pos: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
        """Get valid neighbors of a grid position."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = pos[0] + dx, pos[1] + dy
                if 0 <= new_x < width and 0 <= new_y < height:
                    neighbors.append((new_x, new_y))
        return neighbors
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int], 
                         start: Tuple[int, int]) -> Path:
        """Reconstruct path from A* search."""
        path_nodes = []
        
        while current in came_from:
            path_nodes.append(Node(position=self._grid_to_world(current)))
            current = came_from[current]
        
        path_nodes.append(Node(position=self._grid_to_world(start)))
        path_nodes.reverse()
        
        return Path(nodes=path_nodes)
