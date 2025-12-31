import gym_donkeycar.envs  # Must import .envs to register environments
import gymnasium as gym
import numpy as np
import os

# =================================================================
# 1. Custom Wrapper (Old Gym -> New Gymnasium 변환)
# =================================================================
from gymnasium import spaces

class DonkeyGymnasiumWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_id, conf):
        # Set environment variables for donkey simulator configuration
        os.environ['DONKEY_SIM_PATH'] = conf.get('exe_path', '')
        os.environ['DONKEY_SIM_PORT'] = str(conf.get('port', 9091))
        os.environ['DONKEY_SIM_HEADLESS'] = '0'  # Show UI by default
        
        # 구형 Gym 환경 로드
        import gym as old_gym
        self.env = old_gym.make(env_id)

        # Space 변환
        self.action_space = spaces.Box(
            low=self.env.action_space.low, high=self.env.action_space.high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low, high=self.env.observation_space.high, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return obs, {} # info 추가 반환

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info # terminated, truncated 분리

    def close(self):
        self.env.close()

# =================================================================
# 2. 메인 실행 코드
# =================================================================
if __name__ == "__main__":

    # 🚨 [수정 필수] 본인의 시뮬레이터 경로를 넣어주세요!
    # 보통 .app 우클릭 -> 패키지 내용 보기 -> Contents -> MacOS 안에 실행파일이 있습니다.
    # 예: "/Users/내이름/Downloads/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"
    SIM_PATH = "/Users/youngjunekang/Code/e2e-autonomous-vehicle/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"

    # 경로 존재 여부 확인
    if not os.path.exists(SIM_PATH):
        print(f"❌ 에러: 시뮬레이터 경로가 틀렸습니다!\n경로: {SIM_PATH}")
        exit()

    conf = {
        "exe_path": SIM_PATH,
        "host": "127.0.0.1",
        "port": 9091,
        "body_style": "donkey",
        "body_rgb": (255, 0, 0), # 빨간색 차
        "car_name": "My_First_Bot",
        "font_size": 50
    }

    # 환경 생성 (Wrapper 적용)
    env = DonkeyGymnasiumWrapper("donkey-generated-track-v0", conf=conf)

    print("🚗 시뮬레이터 연결 성공! 주행을 시작합니다...")
    obs, info = env.reset()

    for i in range(1000):
        # 랜덤 액션: [조향(-1~1), 가속(0~1)]
        action = np.array([np.random.uniform(-0.5, 0.5), 0.3])

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print("💥 충돌! 리셋합니다.")
            obs, info = env.reset()

    env.close()
    print("✅ 테스트 완료.")