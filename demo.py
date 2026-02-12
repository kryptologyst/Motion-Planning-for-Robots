"""Main demo script for motion planning."""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import time

from src.motion_planning import (
    RRT, RRTStar, AStar, Environment, Obstacle, 
    MotionPlanningEvaluator, MotionPlanningVisualizer, Config
)


def create_environment(config: Config) -> Environment:
    """Create environment from configuration."""
    obstacles = []
    for obs_data in config.environment.obstacles:
        obstacle = Obstacle(
            position=np.array(obs_data['position']),
            radius=obs_data['radius']
        )
        obstacles.append(obstacle)
    
    return Environment(
        bounds=tuple(config.environment.bounds),
        obstacles=obstacles
    )


def run_single_planning_demo(config: Config) -> None:
    """Run a single planning demonstration."""
    print("Running Single Planning Demo")
    print("=" * 40)
    
    # Create environment
    environment = create_environment(config)
    
    # Create planners
    planners = {
        'RRT': RRT(environment, 
                  max_iterations=config.planner.max_iterations,
                  step_size=config.planner.step_size,
                  goal_tolerance=config.planner.goal_tolerance,
                  random_seed=config.planner.random_seed),
        'RRT*': RRTStar(environment, 
                       max_iterations=config.planner.max_iterations,
                       step_size=config.planner.step_size,
                       goal_tolerance=config.planner.goal_tolerance,
                       rewire_radius=config.planner.rewire_radius,
                       random_seed=config.planner.random_seed),
        'A*': AStar(environment, 
                   resolution=config.planner.resolution,
                   heuristic_weight=config.planner.heuristic_weight)
    }
    
    # Planning parameters
    start = tuple(config.evaluation.start_position)
    goal = tuple(config.evaluation.goal_position)
    
    # Visualizer
    visualizer = MotionPlanningVisualizer(environment)
    
    # Plan with each algorithm
    results = {}
    for name, planner in planners.items():
        print(f"\nPlanning with {name}...")
        start_time = time.time()
        path = planner.plan(start, goal)
        planning_time = time.time() - start_time
        
        if path is not None:
            print(f"  Success! Path length: {path.cost:.3f}")
            print(f"  Planning time: {planning_time*1000:.1f} ms")
            print(f"  Path nodes: {len(path.nodes)}")
            results[name] = path
        else:
            print(f"  Failed to find path")
            results[name] = None
    
    # Create comparison plot
    fig = visualizer.plot_comparison(results, "Motion Planning Algorithm Comparison")
    fig.savefig("assets/planning_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to assets/planning_comparison.png")


def run_evaluation_benchmark(config: Config) -> None:
    """Run comprehensive evaluation benchmark."""
    print("\nRunning Evaluation Benchmark")
    print("=" * 40)
    
    # Create environment
    environment = create_environment(config)
    
    # Create planners
    planners = {
        'RRT': RRT(environment, 
                  max_iterations=config.planner.max_iterations,
                  step_size=config.planner.step_size,
                  goal_tolerance=config.planner.goal_tolerance,
                  random_seed=config.planner.random_seed),
        'RRT*': RRTStar(environment, 
                       max_iterations=config.planner.max_iterations,
                       step_size=config.planner.step_size,
                       goal_tolerance=config.planner.goal_tolerance,
                       rewire_radius=config.planner.rewire_radius,
                       random_seed=config.planner.random_seed),
        'A*': AStar(environment, 
                   resolution=config.planner.resolution,
                   heuristic_weight=config.planner.heuristic_weight)
    }
    
    # Evaluation parameters
    start = tuple(config.evaluation.start_position)
    goal = tuple(config.evaluation.goal_position)
    num_runs = config.evaluation.num_runs
    
    # Evaluator
    evaluator = MotionPlanningEvaluator(environment)
    
    # Run evaluation
    print(f"Running {num_runs} evaluations per planner...")
    results = evaluator.compare_planners(planners, start, goal, num_runs)
    
    # Generate leaderboard
    leaderboard = evaluator.generate_leaderboard(results)
    print(f"\n{leaderboard}")
    
    # Save results
    with open("assets/leaderboard.txt", "w") as f:
        f.write(leaderboard)
    
    # Create metrics plot
    visualizer = MotionPlanningVisualizer(environment)
    fig = visualizer.plot_metrics(results)
    fig.savefig("assets/metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nMetrics plot saved to assets/metrics_comparison.png")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Motion Planning Demo")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                       help="Configuration file path")
    parser.add_argument("--demo", action="store_true", 
                       help="Run single planning demo")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run evaluation benchmark")
    parser.add_argument("--all", action="store_true",
                       help="Run both demo and benchmark")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Create assets directory
    Path("assets").mkdir(exist_ok=True)
    
    # Set random seed for reproducibility
    if config.planner.random_seed is not None:
        np.random.seed(config.planner.random_seed)
    
    # Run requested tasks
    if args.demo or args.all:
        run_single_planning_demo(config)
    
    if args.benchmark or args.all:
        run_evaluation_benchmark(config)
    
    if not (args.demo or args.benchmark or args.all):
        print("No task specified. Use --demo, --benchmark, or --all")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
