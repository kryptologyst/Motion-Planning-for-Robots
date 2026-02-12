"""Visualization tools for motion planning."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.animation as animation
from matplotlib.patches import Circle

from .environment import Environment, Obstacle
from .utils import Path, Node
from .planners import MotionPlanner


class MotionPlanningVisualizer:
    """Visualizer for motion planning algorithms and results."""
    
    def __init__(self, environment: Environment, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize visualizer.
        
        Args:
            environment: Planning environment
            figsize: Figure size for plots
        """
        self.environment = environment
        self.figsize = figsize
        self.setup_style()
    
    def setup_style(self) -> None:
        """Setup matplotlib style."""
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 12
    
    def plot_environment(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot the environment with obstacles and bounds.
        
        Args:
            ax: Matplotlib axes to plot on
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot bounds
        x_min, y_min, x_max, y_max = self.environment.bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Plot obstacles
        for obstacle in self.environment.obstacles:
            circle = Circle(obstacle.position, obstacle.radius, 
                          color='red', alpha=0.7, label='Obstacle')
            ax.add_patch(circle)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Motion Planning Environment')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return ax
    
    def plot_path(self, path: Path, ax: Optional[plt.Axes] = None, 
                  color: str = 'blue', linewidth: float = 2.0,
                  label: str = 'Path') -> plt.Axes:
        """
        Plot a planned path.
        
        Args:
            path: Path to plot
            ax: Matplotlib axes to plot on
            color: Path color
            linewidth: Path line width
            label: Path label
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            ax = self.plot_environment()
        
        if len(path.nodes) < 2:
            return ax
        
        # Extract positions
        positions = np.array([node.position for node in path.nodes])
        
        # Plot path
        ax.plot(positions[:, 0], positions[:, 1], 
               color=color, linewidth=linewidth, label=label)
        
        # Plot start and goal
        ax.scatter(positions[0, 0], positions[0, 1], 
                  color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1], 
                  color='red', s=100, marker='*', label='Goal', zorder=5)
        
        return ax
    
    def plot_tree(self, tree: List[Node], ax: Optional[plt.Axes] = None,
                  color: str = 'lightblue', alpha: float = 0.5) -> plt.Axes:
        """
        Plot the exploration tree.
        
        Args:
            tree: List of nodes in the tree
            ax: Matplotlib axes to plot on
            color: Tree color
            alpha: Tree transparency
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            ax = self.plot_environment()
        
        # Plot tree edges
        for node in tree:
            if node.parent is not None:
                ax.plot([node.parent.position[0], node.position[0]],
                       [node.parent.position[1], node.position[1]],
                       color=color, alpha=alpha, linewidth=1)
        
        # Plot tree nodes
        positions = np.array([node.position for node in tree])
        ax.scatter(positions[:, 0], positions[:, 1], 
                  color=color, s=20, alpha=alpha)
        
        return ax
    
    def plot_comparison(self, results: Dict[str, Path], 
                       title: str = "Motion Planning Comparison") -> plt.Figure:
        """
        Plot comparison of multiple planning results.
        
        Args:
            results: Dictionary of planner names to paths
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax = self.plot_environment(ax)
        
        colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink']
        
        for i, (name, path) in enumerate(results.items()):
            if path is not None:
                color = colors[i % len(colors)]
                ax = self.plot_path(path, ax, color=color, label=name)
        
        ax.legend()
        ax.set_title(title)
        
        return fig
    
    def animate_planning(self, planner: MotionPlanner, 
                        start: Tuple[float, float],
                        goal: Tuple[float, float],
                        save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Create animation of planning process.
        
        Args:
            planner: Motion planner
            start: Start position
            goal: Goal position
            save_path: Path to save animation
            
        Returns:
            Matplotlib animation
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax = self.plot_environment(ax)
        
        # This is a simplified animation - in practice, you'd need to modify
        # the planner to yield intermediate states
        def animate(frame):
            ax.clear()
            ax = self.plot_environment(ax)
            
            # Plot start and goal
            ax.scatter(start[0], start[1], color='green', s=100, marker='o', label='Start')
            ax.scatter(goal[0], goal[1], color='red', s=100, marker='*', label='Goal')
            
            if frame > 0:
                # Simulate planning progress
                path = planner.plan(start, goal)
                if path is not None:
                    ax = self.plot_path(path, ax, color='blue', label='Path')
            
            ax.legend()
            ax.set_title(f'Planning Progress - Frame {frame}')
        
        anim = animation.FuncAnimation(fig, animate, frames=10, interval=500, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
        
        return anim
    
    def plot_metrics(self, results: Dict[str, Dict[str, float]]) -> plt.Figure:
        """
        Plot evaluation metrics comparison.
        
        Args:
            results: Evaluation results from evaluator
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        planners = list(results.keys())
        
        # Success rate
        success_rates = [results[p]['success_rate'] for p in planners]
        axes[0, 0].bar(planners, success_rates, color='green', alpha=0.7)
        axes[0, 0].set_title('Success Rate')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        
        # Path length
        path_lengths = [results[p]['avg_path_length'] for p in planners]
        axes[0, 1].bar(planners, path_lengths, color='blue', alpha=0.7)
        axes[0, 1].set_title('Average Path Length')
        axes[0, 1].set_ylabel('Path Length')
        
        # Planning time
        planning_times = [results[p]['avg_planning_time'] * 1000 for p in planners]
        axes[1, 0].bar(planners, planning_times, color='orange', alpha=0.7)
        axes[1, 0].set_title('Average Planning Time')
        axes[1, 0].set_ylabel('Time (ms)')
        
        # Path smoothness
        smoothness = [results[p]['avg_smoothness'] for p in planners]
        axes[1, 1].bar(planners, smoothness, color='purple', alpha=0.7)
        axes[1, 1].set_title('Average Path Smoothness')
        axes[1, 1].set_ylabel('Smoothness')
        
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution for saved image
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
