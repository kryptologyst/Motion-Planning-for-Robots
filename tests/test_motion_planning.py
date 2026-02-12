"""Test suite for motion planning."""

import unittest
import numpy as np
from src.motion_planning import (
    RRT, RRTStar, AStar, Environment, Obstacle, 
    Node, Path, distance, MotionPlanningEvaluator
)


class TestEnvironment(unittest.TestCase):
    """Test cases for Environment class."""
    
    def setUp(self):
        """Set up test environment."""
        self.obstacles = [
            Obstacle(position=np.array([1.0, 1.0]), radius=0.2),
            Obstacle(position=np.array([2.0, 2.0]), radius=0.1)
        ]
        self.environment = Environment(
            bounds=(0.0, 0.0, 3.0, 3.0),
            obstacles=self.obstacles
        )
    
    def test_valid_position(self):
        """Test position validation."""
        # Valid position
        self.assertTrue(self.environment.is_valid_position([0.5, 0.5]))
        
        # Position in obstacle
        self.assertFalse(self.environment.is_valid_position([1.0, 1.0]))
        
        # Position outside bounds
        self.assertFalse(self.environment.is_valid_position([4.0, 4.0]))
    
    def test_valid_path(self):
        """Test path validation."""
        # Valid path
        self.assertTrue(self.environment.is_valid_path([0.0, 0.0], [0.5, 0.5]))
        
        # Path through obstacle
        self.assertFalse(self.environment.is_valid_path([0.0, 0.0], [2.0, 2.0]))


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_distance(self):
        """Test distance calculation."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        expected_distance = 5.0
        self.assertAlmostEqual(distance(p1, p2), expected_distance, places=6)
    
    def test_node_creation(self):
        """Test Node creation."""
        node = Node(position=[1.0, 2.0])
        self.assertEqual(node.position[0], 1.0)
        self.assertEqual(node.position[1], 2.0)
        self.assertIsNone(node.parent)
        self.assertEqual(node.cost, 0.0)


class TestRRT(unittest.TestCase):
    """Test cases for RRT planner."""
    
    def setUp(self):
        """Set up test environment and planner."""
        self.environment = Environment(
            bounds=(0.0, 0.0, 3.0, 3.0),
            obstacles=[]
        )
        self.planner = RRT(
            environment=self.environment,
            max_iterations=100,
            step_size=0.1,
            random_seed=42
        )
    
    def test_planning_success(self):
        """Test successful planning."""
        start = (0.0, 0.0)
        goal = (2.0, 2.0)
        
        path = self.planner.plan(start, goal)
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path.nodes), 0)
        self.assertTrue(np.allclose(path.nodes[0].position, start))
        self.assertTrue(np.allclose(path.nodes[-1].position, goal, atol=0.1))


class TestRRTStar(unittest.TestCase):
    """Test cases for RRT* planner."""
    
    def setUp(self):
        """Set up test environment and planner."""
        self.environment = Environment(
            bounds=(0.0, 0.0, 3.0, 3.0),
            obstacles=[]
        )
        self.planner = RRTStar(
            environment=self.environment,
            max_iterations=100,
            step_size=0.1,
            random_seed=42
        )
    
    def test_planning_success(self):
        """Test successful planning."""
        start = (0.0, 0.0)
        goal = (2.0, 2.0)
        
        path = self.planner.plan(start, goal)
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path.nodes), 0)


class TestAStar(unittest.TestCase):
    """Test cases for A* planner."""
    
    def setUp(self):
        """Set up test environment and planner."""
        self.environment = Environment(
            bounds=(0.0, 0.0, 3.0, 3.0),
            obstacles=[]
        )
        self.planner = AStar(
            environment=self.environment,
            resolution=0.1
        )
    
    def test_planning_success(self):
        """Test successful planning."""
        start = (0.0, 0.0)
        goal = (2.0, 2.0)
        
        path = self.planner.plan(start, goal)
        
        self.assertIsNotNone(path)
        self.assertGreater(len(path.nodes), 0)


class TestEvaluator(unittest.TestCase):
    """Test cases for MotionPlanningEvaluator."""
    
    def setUp(self):
        """Set up test environment and evaluator."""
        self.environment = Environment(
            bounds=(0.0, 0.0, 3.0, 3.0),
            obstacles=[]
        )
        self.evaluator = MotionPlanningEvaluator(self.environment)
        self.planner = RRT(
            environment=self.environment,
            max_iterations=50,
            step_size=0.1,
            random_seed=42
        )
    
    def test_evaluation(self):
        """Test planner evaluation."""
        start = (0.0, 0.0)
        goal = (2.0, 2.0)
        
        results = self.evaluator.evaluate_planner(self.planner, start, goal, num_runs=10)
        
        self.assertIn('success_rate', results)
        self.assertIn('avg_path_length', results)
        self.assertIn('avg_planning_time', results)
        self.assertGreaterEqual(results['success_rate'], 0.0)
        self.assertLessEqual(results['success_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()
