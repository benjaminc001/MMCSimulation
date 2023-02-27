# Simulación de ataques FDIA y detección en MMC

## Introducción

El objetivo de esta instancia es migrar el desarrollo de un proyecto de aprendizaje reforzado al lenguaje de programación Python, integrando las utilidades de los *framework* de aprendizaje reforzado, stable-baselines y gym, respectivamente. El proyecto de RL consiste en implementar la simulación de un sistema MMC (Convertidor Modular Multinivel), un dispositivo eléctrico con la capacidad de transportar energía a largas distancias (generalmente en el ámbito HVDC) y con características modulares que le permite al usuario aplicar un control de manera distribuida a cada una de las unidades que componen al sistema. Esta simulación presenta un entorno de aprendizaje por refuerzo, donde los agentes a controlar son un atacante generador de ciberataques tipo *False Data Injection* o FDIA, y un detector de dichos ataques.

Los métodos de entrenamiento a utilizar son, principalmente, TD3 para el atacante y DQN para el detector. Ambos métodos de entrenamiento corresponden a la categoría de *off-policy*, es decir, recurren a la estimación del valor esperado de la recompensa para tomar una acción. En el caso del atacante, se decidió añadir dos métodos de entrenamiento: SAC y PPO. 

## Modo de uso 
Para poder correr los entrenamientos, se debe tener en cuenta la instalación previa de las librerías anteriormente mencionadas, además de paquetes como Numpy, Matplotlib y Pytorch, los cuales sirvieron como apoyo adicional en las implementaciones de los ambientes y la visualización de los resultados. Los archivos .zip de la carpeta "agent-files" contienen los modelos pre-entrenados para atacante y detector, los cuales pueden probarse comentando con "\#" las definiciones de los modelos en los scripts de Python que tienen los sufijos "\_atk" o "\_detector". Los ambientes se encuentran en los archivos "env\_gym" y ""
