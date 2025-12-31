import gym_donkeycar # import old library(will be converted to gymnasium)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.donkey_wrapper import DonkeyGymnasiumWrapper
import config
import os

def main():
    # 1. 환경 생성 (우리가 만든 래퍼 사용)
    env = DonkeyGymnasiumWrapper("donkey-generated-track-v0", conf=config.DONKEY_CONF)

    # 2. 모델 정의 (PPO)
    # CnnPolicy: 이미지를 입력으로 받기 때문에 CNN 사용 필수
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=config.Learning_Rate,
        batch_size=config.Batch_Size,
        tensorboard_log="./logs/"
    )

    # 3. 체크포인트 콜백 (중간 저장)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='ppo_donkey'
    )

    # 4. 학습 시작
    print("🏎️ Training Started with Gymnasium Wrapper...")
    try:
        model.learn(
            total_timesteps=config.Total_Timesteps,
            callback=checkpoint_callback
        )
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted by User")

    # 5. 최종 저장
    model.save("ppo_donkey_final")
    env.close()
    print("✅ Training Finished & Model Saved.")

if __name__ == "__main__":
    main()