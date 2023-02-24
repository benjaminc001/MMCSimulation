""" Ambiente del detector que hereda atributos de la clase Env para compatibilidad con Gym """
from gym import Env,spaces
import matlab.engine

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from numpy.random import default_rng

class Detector(Env):
    def __init__(self,path,fs,model) -> None:
        super(Detector,self).__init__()
        # Variables de la simulación
        self._path = path
        self._fs = fs
        self._model = model
        self._ts = 1/fs # Tiempo de muestreo 
        self.is_done = False 

        # Espacio de observación
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(6,),dtype=np.float32)
        # Acciones: detectar si hay o no un ataque
        self.action_space = spaces.Discrete(2)

        # Conexión con el API de MATLAB 
        self.eng = matlab.engine.start_matlab()
        self.eng.eval(r'addpath({})'.format(self._path))

        # Parámetros de PLECs inicializados
        self.eng.init_avg_v1(nargout=0)
        self.eng.eval("model = '{}'".format(self._model),nargout=0)
        self.eng.eval("Ts = {}".format(self._ts),nargout=0)
        
        # Tiempo de ataque y magnitud del ataque
        self.eng.eval("Ti = 0.5",nargout=0)
        self.eng.eval("Tatk1 = 1.0",nargout=0)
        self.eng.eval("Tatk2 = 1.0",nargout=0)
        self.eng.eval("Tstop = Ti + Tatk1 + Tatk2 + 1.5",nargout=0)
        self.eng.eval("Atk1 = 2.0",nargout=0)
        self.eng.eval("Atk2 = 1.0",nargout=0)
        self.eng.eval("Atk3 = 3.0",nargout=0)


        # Primer step
        self.eng.eval("load_system(model)",nargout=0)
        self.eng.set_param('{}/Temp'.format(self._model),'value',str(0),nargout=0)
        self.eng.set_param('{}/detect'.format(self._model),'value',str(0),nargout=0)
        self.eng.set_param(self._model,'SimulationCommand','start','SimulationCommand','pause',nargout=0)
    
    def reset(self):
        """ Función de reset para el detector """
        # Reset del valor que indica si está completado
        self.is_done = False

        # Reinicia el temporizador y la acción
        self.eng.set_param('{}/Temp'.format(self._model),'value',str(0.2),nargout=0)
        self.eng.set_param('{}/detect'.format(self._model),'value',str(0),nargout=0)

        # Restaura la potencia y cambia la inductancia
        p = 1500 - (random.random() - 0.5)*(1500/10)
        self.eng.eval("P = {}".format(p),nargout=0)
        l = 4.15e-3
        L = l - l*(random.random()/(10-0.05)) 
        self.eng.eval("L = {}".format(L),nargout=0)

        # Reinicia los tiempos de inicio y parada
        ti = random.random()*0.1 + 0.15
        tatk1 = random.random() + 0.35
        tatk2 = random.random() + 0.35
        tstop = random.random() + ti + tatk1 + tatk2 + 0.35
        self.eng.eval("Ti = {}".format(ti),nargout=0)
        self.eng.eval("Tatk1 = {}".format(tatk1),nargout=0)
        self.eng.eval("Tatk2 = {}".format(tatk2),nargout=0)
        self.eng.eval("Tstop = {}".format(tstop),nargout=0)

        # Ataques cambiados
        rng = default_rng()
        numbers = rng.choice(4,size=3,replace=False)
        Atk1 = numbers[0]+ random.random()
        Atk2 = numbers[1] + random.random()
        Atk3 = numbers[2] + random.random()
        self.eng.eval("Atk1 = {}".format(Atk1),nargout=0)
        self.eng.eval("Atk2 = {}".format(Atk2),nargout=0)
        self.eng.eval("Atk3 = {}".format(Atk3),nargout=0)

        # Parar y volver a comenzar
        self.eng.set_param(self._model,'SimulationCommand','start','SimulationCommand','pause',nargout=0)
        self.eng.set_param(self._model,'SimulationCommand','continue',nargout=0)

        # Obtener estados y valor de verdad de término
        state = self.observation_func()
        
        return state

    def step(self,action):
        """ Paso de tiempo en el episodio """
        action = action
        self.eng.set_param('{}/detect'.format(self._model),'value',str(action),nargout=0)

        # Continuar la simualación
        self.eng.set_param(self._model,'SimulationCommand','continue',nargout=0)
        # Actualiza el tiempo actual en el temporizador
        self.eng.set_param('{}/Temp'.format(self._model),'value',str(self.eng.eval("get_param(bdroot, 'SimulationTime')")),nargout=0)
        
        # Función de observación
        observation = self.observation_func()
        is_attacked = self.eng.eval("out.isAttacked(end)")
        d_act = self.eng.eval("out.daction(end,:)")

        reward = self.reward(action,is_attacked,d_act)

        done = self.eng.get_param(self._model,'SimulationStatus') == ('stopped' or 'terminating')

        return observation,reward,done,{}

    def observation_func(self):
        """ Función de observación del detector 
        (vector de 6 componentes) """
        error = np.array(self.eng.eval("out.error(end,:)"))
        index = np.array(self.eng.eval("out.index(end,:)"))

        # Retorna la información como vector 
        return np.concatenate([error[0],index[0]]).astype(np.float32) 

    def reward(self,action,is_attacked,d_action):
        """ Reward de la detección:
        Penaliza el cambio en las acciones y las equivocaciones
        Recompensa las detecciones correctas"""
        is_attacked = int(is_attacked)
        Kn1 = -2
        Kn2 = -5
        Kp1 = 1
        fp = np.bitwise_xor(is_attacked,action)
        tp = np.logical_not(fp).astype(int)
        m = 1 - tp
        rew = Kn1*fp + Kn2*m*d_action + Kp1*tp
        
        return rew

    def get_info(self):
        vout = np.array(self.eng.eval("out.vout.Data"))
        indices = np.array(self.eng.eval("out.index"))
        actions = np.array(self.eng.eval("out.action"))

        return vout, indices, actions

    def plot_vout(self,name="voltaje.png"):
        """ Grafica las variables de voltaje en el tiempo 
            - Tensión real
            - Tensión hackeada
            - Tensión estimada """
        vout,_,__ = self.get_info()
        tout = np.array(self.eng.eval("out.vout.Time"))
        plt.plot(tout,vout[:,0],label="Tensión real")
        plt.plot(tout,vout[:,1],label="Tensión hackeada")
        plt.plot(tout,vout[:,2],label="Tensión estimada")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Voltaje [V]")
        plt.title("Variables de voltaje del sistema")
        plt.legend()
        plt.savefig(name)
        plt.show()

    
    def plot_idx(self,name="ind_residual.png"):
        """ Grafica el índice residual, error de estimación del filtro
            y la cota superior del índice (0.1) """
        _,ind,__ = self.get_info()
        tout = np.array(self.eng.eval("out.vout.Time"))
        plt.plot(tout,ind[:,0],label="Índice residual")
        plt.plot(tout,ind[:,1],label="Cota superior")
        plt.plot(tout,ind[:,2],label="Error de estimación")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Índice residual y error de estimación")
        plt.title("Variables del filtro de Kalman")
        plt.legend()
        plt.savefig(name)
        plt.show()
        
    
    def plot_act(self,name="accion.png"):
        """ Grafica la acción del agente en el tiempo """
        vout,ind,act = self.get_info()
        tout = np.array(self.eng.eval("out.vout.Time"))
        plt.plot(tout,act)
        plt.title("Acción del agente") 
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Acción")
        plt.savefig(name)
        plt.show()
    
    def plot_attk(self,name="ataque.png"):
        """ Grafica los step de ataque """
        atk = np.array(self.eng.eval("out.attack.Data"))
        time = np.array(self.eng.eval("out.attack.Time"))
        plt.plot(time,atk)
        plt.title("Ataques generados en el sistema")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Voltaje [V]")
        plt.savefig(name)
        plt.show()

    def close(self):
        """ Cerrar conexión con el ambiente de MATLAB """
        self.eng.set_param(self._model,'SimulationCommand','stop',nargout=0)
        self.eng.quit()



