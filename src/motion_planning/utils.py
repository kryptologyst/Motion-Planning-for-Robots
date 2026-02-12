"""Core utilities for motion planning."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class Node:
    """Represents a node in the planning tree/graph."""
    
    position: np.ndarray
    parent: Optional[Node] = None
    cost: float = 0.0
    
    def __post_init__(self) -> None:
        """Ensure position is a numpy array."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)


@dataclass
class Path:
    """Represents a planned path."""
    
    nodes: List[Node]
    cost: float = 0.0
    
    def __post_init__(self) -> None:
        """Calculate path cost if not provided."""
        if self.cost == 0.0 and len(self.nodes) > 1:
            self.cost = sum(
                distance(self.nodes[i].position, self.nodes[i + 1].position)
                for i in range(len(self.nodes) - 1)
            )
    
    @property
    def positions(self) -> List[np.ndarray]:
        """Get list of positions along the path."""
        return [node.position for node in self.nodes]
    
    def to_array(self) -> np.ndarray:
        """Convert path to numpy array."""
        return np.array(self.positions)


def distance(p1: Union[np.ndarray, Tuple[float, float]], 
             p2: Union[np.ndarray, Tuple[float, float]]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point as numpy array or tuple
        p2: Second point as numpy array or tuple
        
    Returns:
        Euclidean distance between the points
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.linalg.norm(p1 - p2))


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π].
    
    Args:
        angle: Input angle in radians
        
    Returns:
        Normalized angle in [-π, π]
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def interpolate_path(path: Path, resolution: float = 0.1) -> Path:
    """
    Interpolate path with given resolution.
    
    Args:
        path: Input path
        resolution: Interpolation resolution
        
    Returns:
        Interpolated path with higher resolution
    """
    if len(path.nodes) < 2:
        return path
    
    interpolated_nodes = [path.nodes[0]]
    
    for i in range(len(path.nodes) - 1):
        start = path.nodes[i].position
        end = path.nodes[i + 1].position
        
        dist = distance(start, end)
        num_steps = int(np.ceil(dist / resolution))
        
        for j in range(1, num_steps + 1):
            t = j / num_steps
            pos = start + t * (end - start)
            interpolated_nodes.append(Node(position=pos))
    
    return Path(nodes=interpolated_nodes)
