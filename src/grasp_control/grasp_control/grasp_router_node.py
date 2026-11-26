"""Route a grasp to OpenMANIPULATOR-X or the RB3 dummy adapter."""

import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from grasp_interfaces.msg import GraspExecutionResult
from rclpy.node import Node
from std_msgs.msg import Bool, String


class GraspRouterNode(Node):
    """Coordinate preparation, feasibility routing, and fallback handling."""

    IDLE = 'IDLE'
    PREPARING = 'PREPARING'
    WAITING_POSE = 'WAITING_POSE'
    EXECUTING_OMX = 'EXECUTING_OPENMANIPULATOR'
    ROUTED_DUMMY = 'ROUTED_RB3_DUMMY'

    def __init__(self) -> None:
        super().__init__('grasp_router_node')
        self.declare_parameter('candidate_timeout_sec', 10.0)
        self.declare_parameter('enable_keyboard_menu', False)
        self.candidate_timeout = float(
            self.get_parameter('candidate_timeout_sec').value
        )
        self.enable_keyboard_menu = bool(
            self.get_parameter('enable_keyboard_menu').value
        )
        self.lock = threading.Lock()
        self.state = self.IDLE
        self.waiting_since = 0.0

        self.create_subscription(
            Bool, '/grasp/execute_trigger', self.trigger_cb, 10
        )
        self.create_subscription(
            Bool,
            '/grasp/openmanipulator/ready',
            self.openmanipulator_ready_cb,
            10,
        )
        self.create_subscription(
            PoseStamped,
            '/grasp/base/desired_grasp_pose',
            self.candidate_cb,
            10,
        )
        self.create_subscription(
            GraspExecutionResult,
            '/grasp/openmanipulator/result',
            self.openmanipulator_result_cb,
            10,
        )
        self.create_subscription(
            GraspExecutionResult,
            '/grasp/rb3_dummy/result',
            self.dummy_result_cb,
            10,
        )

        self.prepare_pub = self.create_publisher(
            Bool, '/grasp/openmanipulator/prepare', 10
        )
        self.openmanipulator_execute_pub = self.create_publisher(
            PoseStamped, '/grasp/openmanipulator/execute', 10
        )
        self.dummy_execute_pub = self.create_publisher(
            PoseStamped, '/grasp/rb3_dummy/execute', 10
        )
        self.home_pub = self.create_publisher(
            Bool, '/grasp/openmanipulator/home', 10
        )
        self.status_pub = self.create_publisher(
            String, '/grasp/router_status', 10
        )
        self.timeout_timer = self.create_timer(0.1, self.timeout_cb)
        self.publish_status('IDLE: Waiting for execute trigger')
        if self.enable_keyboard_menu:
            self.keyboard_thread = threading.Thread(
                target=self.keyboard_loop,
                name='grasp-keyboard-menu',
                daemon=True,
            )
            self.keyboard_thread.start()

    @staticmethod
    def write_keyboard_menu(terminal) -> None:
        terminal.write(
            '\n=== Grasp keyboard menu ===\n'
            '1 : Start the complete grasp sequence\n'
            's : Show the current router state\n'
            'h : Show this menu again\n'
            'Ctrl+C : Stop all launched grasp nodes\n'
        )

    def keyboard_loop(self) -> None:
        try:
            with open(
                '/dev/tty', 'r', encoding='utf-8', buffering=1
            ) as tty_input, open(
                '/dev/tty', 'w', encoding='utf-8', buffering=1
            ) as tty_output:
                self.write_keyboard_menu(tty_output)
                while rclpy.ok():
                    tty_output.write('grasp> ')
                    command = tty_input.readline()
                    if not command:
                        return
                    command = command.strip().lower()
                    if command == '1':
                        tty_output.write(
                            'Grasp trigger requested from keyboard.\n'
                        )
                        self.trigger_cb(Bool(data=True))
                    elif command == 's':
                        with self.lock:
                            state = self.state
                        tty_output.write(f'Current router state: {state}\n')
                    elif command == 'h':
                        self.write_keyboard_menu(tty_output)
                    elif command:
                        tty_output.write(
                            f'Unknown command: {command!r}. '
                            'Enter h to show the menu.\n'
                        )
        except OSError as exc:
            self.get_logger().warn(
                'Keyboard menu disabled because /dev/tty is unavailable: '
                f'{exc}'
            )

    def publish_status(self, text: str) -> None:
        self.status_pub.publish(String(data=text))
        self.get_logger().info(f'[ROUTER] {text}')

    def trigger_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        with self.lock:
            if self.state != self.IDLE:
                self.get_logger().warn(
                    f'Router is {self.state}; execute trigger ignored.'
                )
                return
            self.state = self.PREPARING
        self.publish_status('PREPARING: Requesting RECOGNITION pose')
        self.prepare_pub.publish(Bool(data=True))

    def openmanipulator_ready_cb(self, msg: Bool) -> None:
        with self.lock:
            if self.state != self.PREPARING:
                return
            if not msg.data:
                self.state = self.IDLE
                failed = True
            else:
                self.state = self.WAITING_POSE
                self.waiting_since = time.monotonic()
                failed = False
        if failed:
            self.publish_status(
                'ERROR: OpenMANIPULATOR-X could not enter RECOGNITION pose'
            )
        else:
            self.publish_status(
                'WAITING_POSE: Waiting for a fresh stable 6D grasp pose'
            )

    def candidate_cb(self, msg: PoseStamped) -> None:
        with self.lock:
            if self.state != self.WAITING_POSE:
                return
            self.state = self.EXECUTING_OMX
        self.publish_status(
            'CHECKING: Sending desired 6D pose to OpenMANIPULATOR-X'
        )
        self.openmanipulator_execute_pub.publish(msg)

    def openmanipulator_result_cb(
        self,
        msg: GraspExecutionResult,
    ) -> None:
        with self.lock:
            if self.state != self.EXECUTING_OMX:
                return
            if msg.feasible:
                self.state = self.IDLE
                route_dummy = False
            else:
                self.state = self.ROUTED_DUMMY
                route_dummy = True

        residual_deg = msg.orientation_residual_rad * 180.0 / 3.141592653589793
        if route_dummy:
            self.publish_status(
                'NO: OpenMANIPULATOR-X infeasible '
                f'(orientation residual={residual_deg:.1f} deg). '
                'Routing to RB3-730 dummy.'
            )
            self.dummy_execute_pub.publish(msg.desired_pose)
            return

        if msg.success:
            self.publish_status('YES: OpenMANIPULATOR-X grasp succeeded')
        else:
            self.publish_status(
                f'ERROR: OpenMANIPULATOR-X execution failed: {msg.message}'
            )

    def dummy_result_cb(self, msg: GraspExecutionResult) -> None:
        with self.lock:
            if self.state != self.ROUTED_DUMMY:
                return
            self.state = self.IDLE
        self.publish_status(
            'RB3_DUMMY: Desired 6D grasp pose received; '
            'no simulation or physical motion was executed.'
        )

    def timeout_cb(self) -> None:
        with self.lock:
            if self.state != self.WAITING_POSE:
                return
            if time.monotonic() - self.waiting_since < self.candidate_timeout:
                return
            self.state = self.IDLE
        self.publish_status('ERROR: Stable grasp-pose wait timed out')
        self.home_pub.publish(Bool(data=True))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspRouterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
