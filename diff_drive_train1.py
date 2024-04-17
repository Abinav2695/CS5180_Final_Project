import torch
from algorithms.exponential_schedule import ExponentialSchedule
from algorithms.dqn_agent import train_dqn, DQN, train_dqn_batch
from envs.escape_room_env import EscapeRoomEnv


def main():
    env = EscapeRoomEnv()

    # Hyperparameters
    gamma = 0.99
    num_steps = 500_000  # Train for a substantial number of steps
    num_saves = 5  # Save models periodically during training
    replay_size = 200_000  # Size of the replay buffer
    replay_prepopulate_steps = 50_000  # Initial steps to fill the buffer
    batch_size = 64
    exploration = ExponentialSchedule(1.0, 0.05, 100_000)  # Exploration strategy

    # Train the model
    dqn_models, returns, lengths, losses = train_dqn(
        env,
        num_steps,
        num_saves=num_saves,
        replay_size=replay_size,
        replay_prepopulate_steps=replay_prepopulate_steps,
        batch_size=batch_size,
        exploration=exploration,
        gamma=gamma,
    )

    # Check if the models have been saved correctly
    assert len(dqn_models) == num_saves
    assert all(isinstance(value, DQN) for value in dqn_models.values())

    # Save models to disk for later analysis or continued training
    checkpoint = {key: model.custom_dump() for key, model in dqn_models.items()}
    torch.save(checkpoint, f"model_basic.pt")

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
