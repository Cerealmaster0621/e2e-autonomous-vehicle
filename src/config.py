import os

# 시뮬레이터 실행 파일 경로 (본인의 경로로 수정 필수!)
# 예: /Users/username/projects/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim
SIM_PATH="/Users/youngjunekang/Code/e2e-autonomous-vehicle/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"

# 시뮬레이터 설정
DONKEY_CONF = {
    "exe_path": SIM_PATH,
    "host": "127.0.0.1",
    "port": 9091,
    "body_style": "donkey",
    "body_rgb": (255, 165, 0), # 주황색 차
    "car_name": "RL_Racer",
    "font_size": 50
}

# 학습 파라미터
Total_Timesteps = 100000
Learning_Rate = 3e-4
Batch_Size = 128