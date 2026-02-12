"""ROS 2 integration for motion planning."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped, PoseArray
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Header
from std_srvs.srv import Empty
import numpy as np
from typing import List, Optional, Tuple

from .planners import RRT, RRTStar, AStar
from .environment import Environment, Obstacle
from .utils import Path as MotionPath


class MotionPlanningNode(Node):
    """ROS 2 node for motion planning services."""
    
    def __init__(self):
        """Initialize the motion planning node."""
        super().__init__('motion_planning_node')
        
        # Create environment
        self.environment = self._create_environment()
        
        # Create planners
        self.planners = {
            'rrt': RRT(self.environment, max_iterations=1000, step_size=0.1),
            'rrt_star': RRTStar(self.environment, max_iterations=1000, step_size=0.1),
            'a_star': AStar(self.environment, resolution=0.1)
        }
        
        # Publishers
        self.path_publisher = self.create_publisher(
            NavPath, 'planned_path', 10
        )
        self.obstacles_publisher = self.create_publisher(
            PoseArray, 'obstacles', 10
        )
        
        # Services
        self.plan_service = self.create_service(
            Empty, 'plan_path', self.plan_path_callback
        )
        
        # Parameters
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 2.0)
        self.declare_parameter('planner_type', 'rrt')
        
        # Timer for publishing obstacles
        self.timer = self.create_timer(1.0, self.publish_obstacles)
        
        self.get_logger().info('Motion Planning Node initialized')
    
    def _create_environment(self) -> Environment:
        """Create the planning environment."""
        obstacles = [
            Obstacle(position=np.array([0.5, 0.5]), radius=0.1),
            Obstacle(position=np.array([1.5, 1.5]), radius=0.1),
            Obstacle(position=np.array([2.0, 1.0]), radius=0.1),
        ]
        
        return Environment(
            bounds=(0.0, 0.0, 3.0, 3.0),
            obstacles=obstacles
        )
    
    def plan_path_callback(self, request, response):
        """Callback for path planning service."""
        try:
            # Get parameters
            start_x = self.get_parameter('start_x').value
            start_y = self.get_parameter('start_y').value
            goal_x = self.get_parameter('goal_x').value
            goal_y = self.get_parameter('goal_y').value
            planner_type = self.get_parameter('planner_type').value
            
            start = (start_x, start_y)
            goal = (goal_x, goal_y)
            
            # Plan path
            planner = self.planners.get(planner_type)
            if planner is None:
                self.get_logger().error(f'Unknown planner type: {planner_type}')
                return response
            
            path = planner.plan(start, goal)
            
            if path is not None:
                # Publish path
                self.publish_path(path)
                self.get_logger().info(f'Path planned successfully with {planner_type}')
            else:
                self.get_logger().warn('No path found')
            
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error in path planning: {str(e)}')
            return response
    
    def publish_path(self, path: MotionPath) -> None:
        """Publish planned path as ROS message."""
        nav_path = NavPath()
        nav_path.header = Header()
        nav_path.header.stamp = self.get_clock().now().to_msg()
        nav_path.header.frame_id = 'map'
        
        for node in path.nodes:
            pose_stamped = PoseStamped()
            pose_stamped.header = nav_path.header
            pose_stamped.pose.position.x = float(node.position[0])
            pose_stamped.pose.position.y = float(node.position[1])
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            
            nav_path.poses.append(pose_stamped)
        
        self.path_publisher.publish(nav_path)
    
    def publish_obstacles(self) -> None:
        """Publish obstacles as ROS message."""
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'
        
        for obstacle in self.environment.obstacles:
            pose_stamped = PoseStamped()
            pose_stamped.header = pose_array.header
            pose_stamped.pose.position.x = float(obstacle.position[0])
            pose_stamped.pose.position.y = float(obstacle.position[1])
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            
            pose_array.poses.append(pose_stamped.pose)
        
        self.obstacles_publisher.publish(pose_array)


def main(args=None):
    """Main function for ROS 2 node."""
    rclpy.init(args=args)
    
    node = MotionPlanningNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
