import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from env_gym import MMCEnv

def main():
    env = MMCEnv(r"addpath('C:\Users\ben19\Desktop\practica-python\sim-files')", 800, "MMC_avg_v3")
    check_env(env)


if __name__ == "__main__":
    main()