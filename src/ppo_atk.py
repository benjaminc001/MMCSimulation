""" Entrenamiento PPO para agente atacante """
# Imports
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env_gym import MMCEnv
import matplotlib.pyplot as plt
import numpy as np
import torch
import time 

# Red neuronal de la política
policy_kwargs = dict(activation_fn=torch.nn.ReLU,net_arch=dict(pi=[32,64,32],vf=[64,128,64]))


def main():
    # Desviación estándar para ruido de exploración
    std = np.sqrt(0.1)
    # Contador de tiempo de ejecución
    exc_time = time.perf_counter()
    # Ambiente inicializado (Ts = 0.0013 [s])
    env = MMCEnv(r"addpath('C:\Users\ben19\Desktop\practica-python\sim-files')", 800, "MMC_avg_v3")
    model = PPO("MlpPolicy",env,learning_rate=1e-3,batch_size=128,policy_kwargs=policy_kwargs)
    # Aprende 1000 time-steps
    model.learn(total_timesteps=1000)
    # Guarda los pesos asociados en archivo .zip
    model.save("ppo-mmc")
    del model

    model = PPO.load("ppo-mmc",env=env)
    # Se pueden consultar las estadísticas de la recompensa entregada en n_eval_episodes
    mean_reward, std_reward = evaluate_policy(model,model.get_env(),n_eval_episodes=10)
    obs = env.reset()
    # Listas para graficar
    observations = []
    rewards = []
    steps = []
    done = False
    i = 0
    # Episodio de prueba de modelo
    step = 0.2
    while not done:
        action,state = model.predict(obs)
        obs,reward,done,info =env.step(action)
        observations.append(obs)
        rewards.append(reward)
        steps.append(step)
        vout,indices,actions = env.get_info()
        print(action)
        step += 1.0/8000
        i+=1
    vout,indices,actions = env.get_info()
    # Voltaje 
    plt.plot(vout)
    plt.title("Voltaje en la red trifásica")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    plt.savefig("vout_ppo.png")
    plt.show()
    # Índice residual 
    plt.plot(indices)
    plt.title("Índices residuales")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Valor de índice residual")
    plt.savefig("idx_ppo.png")
    plt.show()
    # Acciones
    plt.plot(actions)
    plt.title("Acción de agente")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    plt.savefig("acc_ppo.png")
    plt.show()
    # Recompensa
    plt.plot(steps,rewards)
    plt.title("Recompensa en simulación")
    plt.xlabel("Paso")
    plt.ylabel("Recompensa")
    plt.savefig("rew_ppo.png")
    plt.show()
    
    env.close()

    print(f"Tiempo de ejecución: {time.perf_counter()-exc_time} segundos")
if __name__ == "__main__":
    main()
