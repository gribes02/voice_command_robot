import sys

import rclpy
from builtin_interfaces.msg import Duration
from pynput.keyboard import Key, Listener
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future
from whisper_msgs.action._inference import Inference_FeedbackMessage
from geometry_msgs.msg import Twist

from whisper_msgs.action import Inference


class WhisperOnKey(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name=node_name)

        # whisper
        self.batch_idx = -1
        self.whisper_client = ActionClient(self, Inference, "/whisper/inference")

        while not self.whisper_client.wait_for_server(1):
            self.get_logger().warn(
                f"Waiting for {self.whisper_client._action_name} action server.."
            )
        self.get_logger().info(
            f"Action server {self.whisper_client._action_name} found."
        )

        # publisher
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.key_listener = Listener(on_press=self.on_key)
        self.key_listener.start()

        self.get_logger().info(self.info_string())

    def on_key(self, key: Key) -> None:
        if key == Key.esc:
            self.key_listener.stop()
            rclpy.shutdown()
            return

        if key == Key.space:
            # inference goal
            self.on_space()
            return

    def on_space(self) -> None:
        goal_msg = Inference.Goal()
        goal_msg.max_duration = Duration(sec=1, nanosec=0)
        self.get_logger().info(
            f"Requesting inference for {goal_msg.max_duration.sec} seconds..."
        )
        future = self.whisper_client.send_goal_async(
            goal_msg, feedback_callback=self.on_feedback
        )
        future.add_done_callback(self.on_goal_accepted)

    def on_goal_accepted(self, future: Future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected.")
            return

        self.get_logger().info("Goal accepted.")

        future = goal_handle.get_result_async()
        future.add_done_callback(self.on_done)

    def on_done(self, future: Future) -> None:
        result: Inference.Result = future.result().result
        self.get_logger().info(f"Result: {result.transcriptions}")
        # print(f"Output: {output}")
        output = result.transcriptions[0]
        self.send_velocity_cmds(output)

    def send_velocity_cmds(self, output) -> None:
        """
        Send velocity commands based on the recognized speech output.
        """
        # Convert output to lowercase and remove leading/trailing whitespace and punctuation
        cleaned_output = output.lower().strip(' .!')

        self.get_logger().info(f"Sending velocity commands for {cleaned_output}...")

        # Initialize Twist message
        velocity_cmd = Twist()

        if cleaned_output == "stop":
            velocity_cmd.linear.x = 0.0
            velocity_cmd.angular.z = 0.0
        elif cleaned_output == "forward":
            velocity_cmd.linear.x = 0.25
            velocity_cmd.angular.z = 0.0
        elif cleaned_output == "backward":
            velocity_cmd.linear.x = -0.25
            velocity_cmd.angular.z = 0.0
        elif cleaned_output == "left":
            velocity_cmd.linear.x = 0.0
            velocity_cmd.angular.z = 0.25
        elif cleaned_output == "right":
            velocity_cmd.linear.x = 0.0
            velocity_cmd.angular.z = -0.25
        else:
            self.get_logger().warn(f"Unknown command: {output}")
            return

        # Publish the Twist message
        self.publisher.publish(velocity_cmd)

    def on_feedback(self, feedback_msg: Inference_FeedbackMessage) -> None:
        if self.batch_idx != feedback_msg.feedback.batch_idx:
            print("")
            self.batch_idx = feedback_msg.feedback.batch_idx
        sys.stdout.write("\033[K")
        print(f"{feedback_msg.feedback.transcription}")
        sys.stdout.write("\033[F")

    def info_string(self) -> str:
        return (
            "\n\n"
            "\tStarting demo.\n"
            "\tPress ESC to exit.\n"
            "\tPress space to start listening.\n"
            "\tPress space again to stop listening.\n"
        )


def main(args=None):
    rclpy.init(args=args)
    whisper_on_key = WhisperOnKey(node_name="whisper_on_key")
    rclpy.spin(whisper_on_key)
