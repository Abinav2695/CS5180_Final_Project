from envs.escape_room_env import EscapeRoomEnv
import numpy as np


def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def square_motion():
    env = EscapeRoomEnv()
    env.reset()  # Make sure to reset the environment at the start

    # Define actions for moving straight and turning
    move_straight = [0.1, 0.1]  # Both wheels at half of maximum speed
    turn_right = [0.1, -0.1]  # Left wheel forward, right wheel backward to turn right

    # Define the number of steps to move straight
    steps_straight = 50  # Number of steps to move straight

    try:
        target_angle = 0  # Initialize target angle

        for _ in range(4):  # Repeat four times to complete a square
            # Move straight
            for _ in range(steps_straight):
                env.step(move_straight)
                env.render()

            # Prepare to turn right by 90 degrees
            target_angle -= np.pi / 2  # Increase target angle by 90 degrees
            target_angle = normalize_angle(target_angle)  # Normalize the angle

            # Turn right until the robot orientation is approximately the target angle
            while not np.isclose(env.robot.theta, target_angle, atol=0.05):
                print(
                    f"Target Angle :: {target_angle}   Robot Angle :: {env.robot.theta}"
                )
                env.step(turn_right)
                env.render()

            print(
                f"Robot state: Position ({env.robot.x}, {env.robot.y}), Orientation {env.robot.theta}"
            )

    except KeyboardInterrupt:
        print("Simulation stopped manually.")
    finally:
        env.close()


def main():
    env = EscapeRoomEnv()
    try:
        for _ in range(500):
            action = env.action_space.sample()
            env.step(action)
            env.render()
            print(f"Robot state :: {env.robot.theta}")
    except KeyboardInterrupt:
        print("Simulation stopped manually.")
    finally:
        env.close()


if __name__ == "__main__":
    # main()
    square_motion()
