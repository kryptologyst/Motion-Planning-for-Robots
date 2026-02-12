"""Evaluation metrics and benchmarking for motion planning."""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

from .planners import MotionPlanner
from .environment import Environment
from .utils import Path, distance


@dataclass
class EvaluationResult:
    """Results from evaluating a motion planner."""
    
    success: bool
    path_length: float
    planning_time: float
    path_smoothness: float
    path_nodes: int
    
    def __post_init__(self):
        """Calculate additional metrics."""
        self.efficiency = self.path_length / max(self.path_nodes, 1)


class MotionPlanningEvaluator:
    """Evaluator for motion planning algorithms."""
    
    def __init__(self, environment: Environment):
        """
        Initialize evaluator.
        
        Args:
            environment: Planning environment
        """
        self.environment = environment
    
    def evaluate_planner(self, 
                        planner: MotionPlanner,
                        start: Tuple[float, float],
                        goal: Tuple[float, float],
                        num_runs: int = 100) -> Dict[str, float]:
        """
        Evaluate a motion planner with multiple runs.
        
        Args:
            planner: Motion planner to evaluate
            start: Start position
            goal: Goal position
            num_runs: Number of evaluation runs
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = []
        
        for _ in range(num_runs):
            start_time = time.time()
            path = planner.plan(start, goal)
            planning_time = time.time() - start_time
            
            if path is not None:
                path_length = self._calculate_path_length(path)
                smoothness = self._calculate_smoothness(path)
                
                result = EvaluationResult(
                    success=True,
                    path_length=path_length,
                    planning_time=planning_time,
                    path_smoothness=smoothness,
                    path_nodes=len(path.nodes)
                )
            else:
                result = EvaluationResult(
                    success=False,
                    path_length=float('inf'),
                    planning_time=planning_time,
                    path_smoothness=0.0,
                    path_nodes=0
                )
            
            results.append(result)
        
        return self._aggregate_results(results)
    
    def _calculate_path_length(self, path: Path) -> float:
        """Calculate total path length."""
        if len(path.nodes) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path.nodes) - 1):
            total_length += distance(path.nodes[i].position, path.nodes[i + 1].position)
        
        return total_length
    
    def _calculate_smoothness(self, path: Path) -> float:
        """Calculate path smoothness (inverse of curvature)."""
        if len(path.nodes) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(path.nodes) - 1):
            p1 = path.nodes[i - 1].position
            p2 = path.nodes[i].position
            p3 = path.nodes[i + 1].position
            
            # Calculate curvature using cross product
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                # Use 3D cross product for 2D vectors
                v1_3d = np.array([v1[0], v1[1], 0])
                v2_3d = np.array([v2[0], v2[1], 0])
                curvature = abs(np.cross(v1_3d, v2_3d)[2]) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                curvatures.append(curvature)
        
        if not curvatures:
            return 0.0
        
        # Return inverse of average curvature (higher is smoother)
        avg_curvature = np.mean(curvatures)
        return 1.0 / (avg_curvature + 1e-6)
    
    def _aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Aggregate evaluation results."""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                'success_rate': 0.0,
                'avg_path_length': float('inf'),
                'avg_planning_time': float('inf'),
                'avg_smoothness': 0.0,
                'avg_efficiency': 0.0,
                'std_path_length': 0.0,
                'std_planning_time': 0.0,
            }
        
        # Calculate standard deviations safely
        path_lengths = [r.path_length for r in successful_results]
        planning_times = [r.planning_time for r in successful_results]
        
        std_path_length = statistics.stdev(path_lengths) if len(path_lengths) > 1 else 0.0
        std_planning_time = statistics.stdev(planning_times) if len(planning_times) > 1 else 0.0
        
        return {
            'success_rate': len(successful_results) / len(results),
            'avg_path_length': statistics.mean(path_lengths),
            'avg_planning_time': statistics.mean(planning_times),
            'avg_smoothness': statistics.mean([r.path_smoothness for r in successful_results]),
            'avg_efficiency': statistics.mean([r.efficiency for r in successful_results]),
            'std_path_length': std_path_length,
            'std_planning_time': std_planning_time,
        }
    
    def compare_planners(self, 
                        planners: Dict[str, MotionPlanner],
                        start: Tuple[float, float],
                        goal: Tuple[float, float],
                        num_runs: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple motion planners.
        
        Args:
            planners: Dictionary of planner names to planner instances
            start: Start position
            goal: Goal position
            num_runs: Number of evaluation runs per planner
            
        Returns:
            Dictionary of evaluation results for each planner
        """
        results = {}
        
        for name, planner in planners.items():
            print(f"Evaluating {name}...")
            results[name] = self.evaluate_planner(planner, start, goal, num_runs)
        
        return results
    
    def generate_leaderboard(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate a leaderboard from evaluation results.
        
        Args:
            results: Evaluation results from compare_planners
            
        Returns:
            Formatted leaderboard string
        """
        leaderboard = "Motion Planning Algorithm Leaderboard\n"
        leaderboard += "=" * 50 + "\n\n"
        
        # Sort by success rate first, then by path length
        sorted_planners = sorted(
            results.items(),
            key=lambda x: (x[1]['success_rate'], -x[1]['avg_path_length']),
            reverse=True
        )
        
        leaderboard += f"{'Planner':<15} {'Success':<8} {'Path Len':<10} {'Time (ms)':<12} {'Smoothness':<12}\n"
        leaderboard += "-" * 70 + "\n"
        
        for name, metrics in sorted_planners:
            leaderboard += f"{name:<15} {metrics['success_rate']:<8.2f} "
            leaderboard += f"{metrics['avg_path_length']:<10.3f} "
            leaderboard += f"{metrics['avg_planning_time']*1000:<12.1f} "
            leaderboard += f"{metrics['avg_smoothness']:<12.3f}\n"
        
        return leaderboard
