""" Test de ambiente para 10 episodios utilizando acciones random del espacio """

import stable_baselines3
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from env_gym import MMCEnv
import torch
import time 

policy_kwargs = dict(activation_fn=torch.nn.ReLU,net_arch=dict(pi=[32,1024,32],qf=[64,512,64]))


def main():
    exc_time = time.perf_counter()
    env = MMCEnv(r"addpath('C:\Users\ben19\Desktop\practica-python\sim-files')", 8000, "MMC_avg_v3")
    model = TD3("MlpPolicy",env,policy_kwargs=policy_kwargs,batch_size=128,verbose=1)
    model.learn(total_timesteps=1e4)
    model.save("td3-mmc")
    del model

    model = TD3.load("td3-mmc",env=env)
    mean_reward, std_reward = evaluate_policy(model,model.get_env(),n_eval_episodes=10)
    vec_env = model.get_env()
    obs = vec_env.reset()
    episodes = 400
    for episode in range(episodes):
        done = False
        print(f"episode {episode + 1}/{episodes}")
        while not done:
            action,_state = model.predict(obs)
            obs, reward, done, info = vec_env.step(action)
            print("obs: ",obs)
            print("action: ",action[0])
            print("reward: ",reward[0])
            
    print(f"Tiempo de ejecución: {time.perf_counter()-exc_time} segundos")
if __name__ == "__main__":
    main()


