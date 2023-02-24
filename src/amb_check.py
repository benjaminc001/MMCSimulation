import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from env_gym import MMCEnv
from detector import Detector

def main()->None:
    """ Programa principal
        1- Inicializa el ambiente
        2- Utiliza el comando de stable baselines para comprobar el ambiente """
    environment = "detector" # Se puede cambiar
    if environment == "detector":
        env = Detector(r"addpath('C:\Users\ben19\Desktop\practica-python\sim-files')", 800, "MMC_detector")
        check_env(env)
        env.close()
    elif environment == "attacker":
        env = MMCEnv(r"addpath('C:\Users\ben19\Desktop\practica-python\sim-files')", 800, "MMC_avg_v3")
        check_env(env)
        env.close()
    else:
        raise Exception("Ingrese un nombre de ambiente válido")

if __name__ == "__main__":
    main()