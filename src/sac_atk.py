""" Entrenamiento SAC para agente atacante """
# Imports
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from env_gym import MMCEnv
import matplotlib.pyplot as plt
import numpy as np
import time 


def main():
    # Contador de tiempo de ejecución
    exc_time = time.perf_counter()
    # Ambiente inicializado (Ts = 0.0013 [s])
    env = MMCEnv(r"addpath('C:\Users\ben19\Desktop\practica-python\sim-files')", 800, "MMC_avg_v3")
    model = SAC("MlpPolicy",env)
    # Aprende 1000 time-steps
    model.learn(total_timesteps=1000)
    # Guarda los pesos asociados en archivo .zip
    model.save("sac-mmc")
    del model

    model = SAC.load("sac-mmc",env=env)
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
    env.plot_vout("vout_sac.png")
    env.plot_idx("idx_sac.png")
    env.plot_act("acc_sac.png")
    # Recompensa
    plt.plot(steps,rewards)
    plt.title("Recompensa en simulación")
    plt.xlabel("Paso")
    plt.ylabel("Recompensa")
    plt.savefig("rew_sac.png")
    plt.show()
    
    env.close()

    print(f"Tiempo de ejecución: {time.perf_counter()-exc_time} segundos")
if __name__ == "__main__":
    main()
