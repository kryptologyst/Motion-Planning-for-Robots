"""Configuration management for motion planning."""

from __future__ import annotations

import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PlannerConfig:
    """Configuration for motion planners."""
    
    # Common parameters
    max_iterations: int = 1000
    step_size: float = 0.1
    goal_tolerance: float = 0.1
    random_seed: Optional[int] = None
    
    # RRT* specific
    rewire_radius: float = 0.5
    
    # A* specific
    resolution: float = 0.1
    heuristic_weight: float = 1.0


@dataclass
class EnvironmentConfig:
    """Configuration for environment."""
    
    bounds: tuple = (0.0, 0.0, 3.0, 3.0)  # x_min, y_min, x_max, y_max
    obstacles: list = None
    
    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = [
                {"position": [0.5, 0.5], "radius": 0.1},
                {"position": [1.5, 1.5], "radius": 0.1},
                {"position": [2.0, 1.0], "radius": 0.1},
            ]


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    num_runs: int = 100
    start_position: tuple = (0.0, 0.0)
    goal_position: tuple = (2.0, 2.0)
    metrics: list = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "success_rate",
                "path_length",
                "planning_time",
                "path_smoothness",
            ]


@dataclass
class Config:
    """Main configuration class."""
    
    planner: PlannerConfig
    environment: EnvironmentConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> Config:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration object
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            planner=PlannerConfig(**data.get('planner', {})),
            environment=EnvironmentConfig(**data.get('environment', {})),
            evaluation=EvaluationConfig(**data.get('evaluation', {}))
        )
    
    def to_yaml(self, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML configuration file
        """
        data = {
            'planner': asdict(self.planner),
            'environment': asdict(self.environment),
            'evaluation': asdict(self.evaluation)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
