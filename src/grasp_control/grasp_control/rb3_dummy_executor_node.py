"""Dummy RB3-730 adapter used until the simulator is implemented."""

import rclpy
from geometry_msgs.msg import PoseStamped
from grasp_interfaces.msg import GraspExecutionResult
from rclpy.node import Node


class Rb3DummyExecutorNode(Node):
    """Accept and report a routed 6D pose without moving anything."""

    def __init__(self) -> None:
        super().__init__('rb3_dummy_executor_node')
        self.create_subscription(
            PoseStamped,
            '/grasp/rb3_dummy/execute',
            self.execute_cb,
            10,
        )
        self.result_pub = self.create_publisher(
            GraspExecutionResult,
            '/grasp/rb3_dummy/result',
            10,
        )
        self.get_logger().info('RB3-730 dummy executor ready.')

    def execute_cb(self, desired_pose: PoseStamped) -> None:
        position = desired_pose.pose.position
        orientation = desired_pose.pose.orientation
        self.get_logger().warn(
            'RB3-730 DUMMY ROUTE: '
            f'frame={desired_pose.header.frame_id}, '
            f'position=({position.x:.3f}, {position.y:.3f}, '
            f'{position.z:.3f}), '
            f'quaternion=({orientation.x:.4f}, {orientation.y:.4f}, '
            f'{orientation.z:.4f}, {orientation.w:.4f}). '
            'No robot or simulator motion was executed.'
        )
        result = GraspExecutionResult()
        result.header = desired_pose.header
        result.robot_name = 'rb3_730_dummy'
        result.feasible = True
        result.executed = False
        result.success = False
        result.desired_pose = desired_pose
        result.command_pose = desired_pose
        result.message = (
            'RB3-730 dummy accepted the route; execution is not implemented.'
        )
        self.result_pub.publish(result)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Rb3DummyExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
