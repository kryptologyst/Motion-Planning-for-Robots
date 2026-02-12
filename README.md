# Motion Planning for Robots

A comprehensive motion planning library implementing state-of-the-art algorithms for robotics applications.

## Features

- **Multiple Planning Algorithms**: RRT, RRT*, A*, and more
- **Comprehensive Evaluation**: Success rate, path length, planning time, smoothness metrics
- **Interactive Visualization**: Real-time plotting and comparison tools
- **Configurable Environment**: Easy setup of obstacles and bounds
- **ROS 2 Integration**: Ready for real robot deployment
- **Extensive Testing**: Full test coverage with pytest

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Motion-Planning-for-Robots.git
cd Motion-Planning-for-Robots

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.motion_planning import RRT, Environment, Obstacle

# Create environment
obstacles = [
    Obstacle(position=[1.0, 1.0], radius=0.2),
    Obstacle(position=[2.0, 2.0], radius=0.1)
]
env = Environment(bounds=(0.0, 0.0, 3.0, 3.0), obstacles=obstacles)

# Create planner
planner = RRT(environment=env, max_iterations=1000, step_size=0.1)

# Plan path
start = (0.0, 0.0)
goal = (2.5, 2.5)
path = planner.plan(start, goal)

if path:
    print(f"Path found! Length: {path.cost:.3f}")
else:
    print("No path found")
```

### Running Demos

```bash
# Run single planning demo
python demo.py --demo

# Run evaluation benchmark
python demo.py --benchmark

# Run both demos
python demo.py --all
```

## Algorithms

### RRT (Rapidly-exploring Random Tree)
- Probabilistic sampling-based planner
- Good for high-dimensional spaces
- Non-optimal but fast

### RRT*
- Optimal variant of RRT
- Rewires tree for better paths
- Asymptotically optimal

### A* (A-Star)
- Grid-based optimal planner
- Uses heuristic for efficiency
- Guaranteed optimal on grid

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
planner:
  max_iterations: 1000
  step_size: 0.1
  goal_tolerance: 0.1
  random_seed: 42

environment:
  bounds: [0.0, 0.0, 3.0, 3.0]
  obstacles:
    - position: [1.0, 1.0]
      radius: 0.2

evaluation:
  num_runs: 100
  start_position: [0.0, 0.0]
  goal_position: [2.0, 2.0]
```

## Evaluation Metrics

- **Success Rate**: Percentage of successful planning attempts
- **Path Length**: Total distance of planned path
- **Planning Time**: Time required to find a path
- **Path Smoothness**: Measure of path curvature
- **Efficiency**: Path length per node ratio

## Project Structure

```
motion-planning/
├── src/motion_planning/          # Main library code
│   ├── __init__.py
│   ├── planners.py               # Planning algorithms
│   ├── environment.py            # Environment representation
│   ├── utils.py                  # Utility functions
│   ├── config.py                 # Configuration management
│   ├── evaluation.py             # Evaluation framework
│   └── visualization.py          # Visualization tools
├── config/                       # Configuration files
│   └── default.yaml
├── tests/                        # Test suite
│   └── test_motion_planning.py
├── assets/                       # Generated outputs
├── demo.py                       # Main demo script
├── pyproject.toml               # Project configuration
└── README.md                     # This file
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/motion_planning

# Run specific test
pytest tests/test_motion_planning.py::TestRRT
```

## ROS 2 Integration

For real robot deployment, install ROS 2 dependencies:

```bash
pip install -e ".[ros2]"
```

The library provides ROS 2 nodes for:
- Path planning service
- Environment visualization
- Real-time planning updates

## Safety Disclaimer

**IMPORTANT**: This software is designed for research and educational purposes only. 

- Do not use on real robots without expert review and safety measures
- Always test in simulation before hardware deployment
- Implement proper safety guards, emergency stops, and velocity limits
- Verify all paths before execution on physical systems
- This software comes with NO WARRANTY and NO LIABILITY

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{motion_planning_2024,
  title={Motion Planning for Robots: A Comprehensive Library},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Motion-Planning-for-Robots}
}
```

## Acknowledgments

- RRT algorithm by LaValle and Kuffner
- RRT* algorithm by Karaman and Frazzoli
- A* algorithm by Hart, Nilsson, and Raphael
- ROS 2 community for robotics framework
# Motion-Planning-for-Robots
