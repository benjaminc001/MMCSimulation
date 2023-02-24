from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from detector import Detector

import numpy as np
import matplotlib.pyplot as plt
import torch
import time

def main():
    # Tiempo de ejecución
    exc_time = time.perf_counter()
    # Definición de ambiente y modelo DQN
    env = Detector(r"addpath('C:\Users\ben19\Desktop\practica-python\sim-files')", 800, "MMC_detector")
    model = DQN("MlpPolicy",env=env,learning_rate=1e-3,batch_size=128,exploration_fraction=0.01,exploration_final_eps=0.1,verbose=True)
    model.learn(total_timesteps=1e4)
    # Se guarda el modelo en disco y se elimina de la RAM
    model.save("dqn_detector")
    del model
    
    # Cargar el archivo
    model = DQN.load("dqn_detector",env=env)
    # Se pueden consultar las estadísticas de la recompensa entregada en n_eval_episodes
    mean_reward, std_reward = evaluate_policy(model,model.get_env(),n_eval_episodes=10)
    obs = env.reset()
    # Listas para graficar
    observations = []
    rewards = []
    steps = []
    done = False
    # Episodio de prueba de modelo
    step = 0.2
    while not done:
        action,state = model.predict(obs)
        obs,reward,done,info =env.step(action)
        rewards.append(reward)
        steps.append(step)
        print(action)
        step += 1.0/800
    env.plot_vout("vout_dqn.png")
    env.plot_idx("idx_dqn.png")
    env.plot_act("acc_dqn.png")
    env.plot_attk("step_atk.png")
    # Recompensa
    plt.plot(steps,rewards)
    plt.title("Recompensa en simulación")
    plt.xlabel("Paso")
    plt.ylabel("Recompensa")
    plt.savefig("rew_dqn.png")
    plt.show()
    
    env.close()

    print(f"Tiempo de ejecución: {time.perf_counter()-exc_time} segundos")


if __name__=="__main__":
    main()
