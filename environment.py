import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp

class MissileGuidanceEnv(gym.Env):
    def __init__(self, logging=False):
        super(MissileGuidanceEnv, self).__init__()

        # Simulation parameters 
        self.dt = 0.01  # Fixed time step
        self.VM = 600   # Missile velocity (m/s)
        self.VT = 400   # Target velocity (m/s)
        self.a_max = 100  # Max missile acceleration (m/s²)
        self.time = 0.0

        self.logging = logging
        self.saved_scenario = None

        # Define action and observation space
        self.action_space = spaces.Box(low=-self.a_max, high=self.a_max, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, -1000, -10]),
            high=np.array([50000, np.pi, 1000, 10]),
            dtype=np.float32
        )

        if self.logging:
            self._init_logs()

    def _init_logs(self):
        self.time_log = []
        self.range_log = []
        self.lambda_log = []
        self.r_dot_log = []
        self.lambda_dot_log = []
        self.aM_log = []
        self.gammaM_log = []
        self.gammaT_log = []
        self.xM_log = [0]
        self.yM_log = [0]
        self.xT_log = []
        self.yT_log = []

    def save_current_scenario(self):
        self.saved_scenario = {
            "r": self.r,
            "lambda_": self.lambda_,
            "gammaM": self.gammaM,
            "gammaT": self.gammaT,
            "tau": self.tau,
            "initZEM": self.initial_ZEM
        }

    def reset(self, replay=False):
        if not replay or self.saved_scenario is None:
            # Randomize new scenario
            self.r = np.random.uniform(4000, 6000)
            self.lambda_ = np.radians(np.random.uniform(-10, 10))
            self.gammaM = np.radians(np.random.uniform(0, 20))
            self.gammaT = np.radians(np.random.uniform(140, 160))
            self.tau = np.random.uniform(0.1, 0.3)
            _, _, self.initial_ZEM = self._compute_ZEM()
            # Save it for possible replay
            self.save_current_scenario()
        else:
            # Replay the last saved scenario
            self.r = self.saved_scenario["r"]
            self.lambda_ = self.saved_scenario["lambda_"]
            self.gammaM = self.saved_scenario["gammaM"]
            self.gammaT = self.saved_scenario["gammaT"]
            self.tau = self.saved_scenario["tau"]
            self.initial_ZEM = self.saved_scenario["initZEM"]

        self.aM = 0.0
        self.time = 0.0

        if self.logging:
            self._init_logs()

        return self._get_state()
    
    def _compute_ZEM(self):
        Vr = self.VT * np.cos(self.gammaT - self.lambda_) - self.VM * np.cos(self.gammaM - self.lambda_)
        Vlambda = self.VT * np.sin(self.gammaT - self.lambda_) - self.VM * np.sin(self.gammaM - self.lambda_)
        ZEM = self.r * Vlambda / max(np.sqrt(Vr**2 + Vlambda**2), 1e-5)
        return Vr, Vlambda, ZEM

    def step(self, action):
        ac = np.clip(action[0], -self.a_max, self.a_max)  # Clip action to allowable range
        y0 = [self.r, self.lambda_, self.gammaM, self.aM]

        # Define dynamics function
        def dynamics(t, y):
            r, lambda_, gammaM, aM = y

            aM_dot = (-aM + ac) / self.tau
            gammaM_dot = aM / self.VM

            Vr = self.VT * np.cos(self.gammaT - lambda_) - self.VM * np.cos(gammaM - lambda_)
            Vlambda = self.VT * np.sin(self.gammaT - lambda_) - self.VM * np.sin(gammaM - lambda_)

            r_dot = Vr
            lambda_dot = Vlambda / self.r if self.r > 0.01 else Vlambda / 0.01

            return [r_dot, lambda_dot, gammaM_dot, aM_dot]

        # Solve the ODE over one time step
        sol = solve_ivp(dynamics, (0, self.dt), y0, method='RK45', t_eval=[self.dt])
        self.r, self.lambda_, self.gammaM, self.aM = sol.y[:, -1]
        self.time += self.dt

        # If logging enabled, record data
        if self.logging:
            Vr, Vlambda, _ = self._compute_ZEM()

            r_dot = Vr
            lambda_dot = Vlambda / self.r if self.r > 0.01 else Vlambda / 0.01

            self.time_log.append(self.time)
            self.range_log.append(self.r)
            self.lambda_log.append(self.lambda_)
            self.r_dot_log.append(r_dot)
            self.lambda_dot_log.append(lambda_dot)
            self.aM_log.append(self.aM)
            self.gammaM_log.append(self.gammaM)
            self.gammaT_log.append(self.gammaT)

            new_xM = self.xM_log[-1] + self.VM * np.cos(self.gammaM) * self.dt
            new_yM = self.yM_log[-1] + self.VM * np.sin(self.gammaM) * self.dt
            new_xT = new_xM + self.r * np.cos(self.lambda_)
            new_yT = new_yM + self.r * np.sin(self.lambda_)

            # new_xM = self.xM_log[-1] + self.VM * np.cos(self.gammaM-self.lambda_) * self.dt
            # new_yM = self.yM_log[-1] + self.VM * np.sin(self.gammaM-self.lambda_) * self.dt
            # new_xT = new_xM + self.r
            # new_yT = new_yM

            self.xM_log.append(new_xM)
            self.yM_log.append(new_yM)
            self.xT_log.append(new_xT)
            self.yT_log.append(new_yT)

        # Get new state
        state = self._get_state()

        # Check termination condition
        done = bool(self.r < 2.0 or self.r > 6000)

        # Calculate reward
        reward = self._compute_reward(ac)

        return state, reward, done, {}

    def _get_state(self):
        Vr, Vlambda, _ = self._compute_ZEM()

        r_dot = Vr
        lambda_dot = Vlambda / self.r if self.r > 0.01 else Vlambda / 0.01

        return np.array([self.r, self.lambda_, r_dot, lambda_dot], dtype=np.float32)

    def _compute_reward(self, ac):
        # Constants from Table 1 in the paper
        ka = -0.2  # ka - acceleration penalty coefficient
        kz = -1    # kz - ZEM penalty coefficient  
        kVr = -2   # kVr - velocity penalty coefficient
        kr = 10    # kr - proximity bonus coefficient
        rd = 20    # rd - proximity threshold (m)
        rdm = 100  # rdm - motivation threshold (m)
        rdmc = 2   # rdmk - motivation coefficient
        
        # Get zero-effort miss (ZEM) and relative velocities
        Vr, _, ZEM = self._compute_ZEM()
        
        # Compute reward components based on Equations 33-36
        ra = ka * (self.aM / self.a_max)**2  # Equation 33 - acceleration cost
        rz = kz * (ZEM / max(abs(self.initial_ZEM), 1e-5))**2  # Equation 34 - ZEM cost
        
        # rVr component from Equation 35
        if Vr > 0:
            rVr = kVr
        else:
            rVr = 0
            
        # rr component from Equation 36
        if self.r <= rd:
            rr = kr
        elif self.r <= rdm:
            rr = rdmc
        else:
            rr = 0
        
        # Total reward from Equation 32
        reward = ra + rz + rVr + rr
        
        return reward

import matplotlib.pyplot as plt

def plot_trajectories(env):
    """
    Plots missile and target trajectories, range vs time, and acceleration vs time.
    
    Args:
        env: The MissileGuidanceEnv instance (must have logging enabled)
    """

    if not env.logging:
        print("Logging was disabled. No data to plot!")
        return

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # 1. Plot missile and target paths (x vs y)
    axs[0].plot(env.xM_log, env.yM_log, label='Missile Path', color='blue')
    axs[0].plot(env.xT_log, env.yT_log, label='Target Path', color='red', linestyle='--')
    axs[0].set_xlabel('X Position (m)')
    axs[0].set_ylabel('Y Position (m)')
    axs[0].set_title('Missile and Target Trajectories')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')  # Equal scaling for x and y

    # 2. Plot Range vs Time
    axs[1].plot(env.time_log, env.range_log, color='green')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Range (m)')
    axs[1].set_title('Range vs Time')
    axs[1].grid(True)

    # 3. Plot Acceleration vs Time
    axs[2].plot(env.time_log, env.aM_log, color='purple')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Missile Lateral Acceleration (m/s²)')
    axs[2].set_title('Missile Acceleration vs Time')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
