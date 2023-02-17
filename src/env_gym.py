# Imports de librerías esenciales
import matlab.engine
import numpy as np
import time 
import gym
from gym import Env,spaces

# Definición de clase de ambiente (hereda atributos de gym.Env)
class MMCEnv(Env):
    def __init__(self,path,fs,model,knowledge="system"):
        super(MMCEnv,self).__init__()
        
        # Variables de simulación
        self._path = path
        self._fs = fs
        self._ts = 1/fs
        self._model = model
        self.knowledge = knowledge
        
        # Variables de RL
        self.reward = None
        self.obs = None
        self.actions = None
        self.is_attacked = False
        self.is_done = False 

        #self.ex_time = time.perf_counter()

        # Inicializa MATLAB
        self.eng = matlab.engine.start_matlab()
        self.eng.eval(r'addpath({})'.format(self._path))

        if self.knowledge == "system":
            self.obs_shape = 5
            self.obs_func = self.system_obs
        elif self.knowledge == "detector":
            self.obs_shape = 6
            self.obs_func = self.detector_obs
        else:
            raise Exception("Ingrese un tipo de observación válido")

        # Espacios de observación y acción para el atacante        
        self.obs_low = np.repeat(np.array([-np.inf]),self.obs_shape) # Por ahora no hay una cota definida 
        self.obs_high = np.repeat(np.array([np.inf]),self.obs_shape)
        self.observation_space = spaces.Box(low=self.obs_low,high=self.obs_high,shape=(self.obs_shape,),dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-50.0]),high=np.array([50.0]),shape=(1,),dtype=np.float32)
        
        # Parámetros de PLECs inicializados
        self.eng.init_avg_v1(nargout=0)
        self.eng.eval("model = '{}'".format(self._model),nargout=0)
        #self.eng.set_param('{}/Ts'.format(self._model),'value',str(self._ts),nargout=0)
        self.eng.eval("Ts = {}".format(self._ts),nargout=0)

        # Primer step
        self.eng.eval("load_system(model)",nargout=0)
        self.eng.set_param('{}/Temp'.format(self._model),'value',str(0),nargout=0)
        self.eng.set_param('{}/action'.format(self._model), 'value', str(0), nargout=0)
        self.eng.set_param(self._model,'SimulationCommand','start','SimulationCommand','pause',nargout=0)
    
        #self.init_time = time.perf_counter() - self.ex_time
        #print(f"Tiempo de inicialización: {self.init_time} segundos")
    def reset(self):
        """ Función de reset para el atacante """
        # Valores de verdad del ambiente se resetean
        self.is_attacked = False
        self.is_done = False

        # Inicializa el temporizador y la acción
        self.eng.set_param('{}/Temp'.format(self._model),'value',str(0.2),nargout=0)
        self.eng.set_param('{}/action'.format(self._model),'value',str(0),nargout=0)
    
        # Restaura la potencia (parámetro de PLECs)
        p = 1500 -(np.random.rand() - 0.5)*(1500/10)
        self.eng.eval("P = {}".format(p),nargout=0)

        # Parar la simulación y volver a iniciar
        self.eng.set_param(self._model,'SimulationCommand','start','SimulationCommand','pause',nargout=0)
        self.eng.set_param(self._model,'SimulationCommand','continue',nargout=0)

        # Obtener estados y término
        state = self.obs_func()

        return state
    
    def step(self, action):
        #t_step = time.perf_counter()
        """ Paso de tiempo en el episodio """
        action = action[0] # Convertir en float
        self.eng.set_param('{}/action'.format(self._model),'value',str(action),nargout=0)


        # Continúa la simulación
        self.eng.set_param(self._model,'SimulationCommand','continue', nargout=0)
        # Actualiza el tiempo actual en el temporizador
        self.eng.set_param('{}/Temp'.format(self._model),'value',str(self.eng.eval("get_param(bdroot, 'SimulationTime')")),nargout=0)
    
        obs = self.obs_func()
        t = self.eng.eval("out.tout(end)")
        idx = self.eng.eval("out.i(end,:)")

        # Reward calculada según la acción tomada
        reward = self.get_reward(obs,t,idx)

        done = self.eng.get_param(self._model,'SimulationStatus') == ('stopped' or 'terminating')
        #tot = time.perf_counter() - t_step
        
        #print(f"Tiempo de ejecución de paso: {tot} segundos")
        return obs,reward,done,{}
    
    def system_obs(self):
        """ Observación cuando se conoce la información del sistema, entrega
        el error, su derivada, integral, la corriente y el índice de modulación  """
        error = np.array(self.eng.eval("out.error(end,:)"))
        i_arm = np.array([self.eng.eval("out.i_aU(end)")])
        m_i = np.array([self.eng.eval("out.m1(end)")])
    
        # Retorna como un vector con la información
        return np.concatenate([error[0],i_arm,m_i]).astype(np.float32)
    
    def detector_obs(self):
        """ Observación cuando se conoce la información del detector """
        error = np.array(self.eng.eval("out.error(end,:)"))
        index = np.array(self.eng.eval("out.index(end,:)"))

        # Retorna la información como vector 
        return np.concatenate([error[0],index[0]]).astype(np.float32)  
    
    def get_reward(self,obs,t,index):
        """ Función de recompensa """
        error = obs[0]
        d_error = obs[1]

        if t<=0.2:
            reward = 0
            self.is_attacked = False
        elif self.is_attacked == True:
            reward = -(error)**2 - (1 + np.abs(d_error))**2
        elif index < 0:
            reward = -(error)**2 - (1 + np.abs(d_error))**2
            self.is_attacked = True
        else:
            reward = +(error)**2 + (1 + np.abs(d_error))**2   
        
        return reward
    
    def get_info(self):
        vout = np.array(self.eng.eval("out.vout"))
        indices = np.array(self.eng.eval("out.index"))
        actions = np.array(self.eng.eval("out.action"))

        return vout, indices, actions
    
    def close(self):
        """ Cerrar conexión con el ambiente de MATLAB """
        self.eng.set_param(self._model,'SimulationCommand','stop',nargout=0)
        self.eng.quit()
