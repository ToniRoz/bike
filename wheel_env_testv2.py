import gymnasium as gym 

from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix

import numpy as np

from numba import njit

"""
Todo:
add new state spaces:
     add code for current (only rimpoints state)
     look up how to calculate the spoketensions (can we jit this as well?)
     add the other two sate space configurations 
     look through the whole code and remove / edit gpt comments
"""


@njit
def fast_wheel_calc(K, F_matrix, B_rad, B_lat, B_tan, tensionchanges, state_space_selection):


    # Compute force vector from tension
    #F = A_adj.dot(tensionchanges)
    F = F_matrix @ tensionchanges

    # Solve for deformation modes
    dm = np.linalg.solve(K, F)
    
    # Compute deflections
    rad_def = B_rad @ dm
    lat_def = B_lat @ dm
    tan_def = B_tan @ dm

    # Stack into (n, 3) deflection array
    n = len(rad_def)
    tot_def = np.empty((n, 3), dtype=np.float64)
    tot_def[:, 0] = rad_def * 1000 # convert to mm from m to avoid very small numbers
    tot_def[:, 1] = lat_def * 1000
    tot_def[:, 2] = tan_def * 1000

    # Compute reward (negative sum of norm per point)
    reward = -np.sum(np.sqrt(tot_def[:, 0]**2 + tot_def[:, 1]**2 + tot_def[:, 2]**2))

    next_state = tot_def.flatten()

    return next_state, reward

@njit
def fast_wheel_calc_with_tension(
    K, F_matrix,
    B_rad, B_lat, B_tan,
    tensionchanges,
    n_vec, b_vec, EA, lengths,
    B_spk
):
    # -------------------------
    # Solve rim deformation
    # -------------------------
    F = F_matrix @ tensionchanges
    dm = np.linalg.solve(K, F)

    rad_def = B_rad @ dm
    lat_def = B_lat @ dm
    tan_def = B_tan @ dm

    # Rim state
    npts = len(rad_def)
    tot_def = np.empty((npts, 3))
    tot_def[:, 0] = rad_def * 1000
    tot_def[:, 1] = lat_def * 1000
    tot_def[:, 2] = tan_def * 1000

    reward = -np.sum(np.sqrt(
        tot_def[:, 0]**2 +
        tot_def[:, 1]**2 +
        tot_def[:, 2]**2
    ))

    # -------------------------
    # Correct tension computation: d = B_theta(θ_spoke) @ dm
    # -------------------------
    n_spokes = len(tensionchanges)
    dT = np.empty(n_spokes)

    for i in range(n_spokes):
        # Compute d vector exactly like ModeMatrix.spoke_tension_change
        d = B_spk[i] @ dm   # (4-element vector: u, v, w, phi)

        u = d[0]
        v = d[1]
        w = d[2]
        phi = d[3]

        # Compute un = (u, v, w) + phi * cross(e3, b)
        cx = -b_vec[i, 2]
        cy =  b_vec[i, 0]
        cz =  0.0

        un0 = u + phi * cx
        un1 = v + phi * cy
        un2 = w + phi * cz

        a = tensionchanges[i]  # adjustment

        dT[i] = EA[i]/lengths[i] * (
            a - (n_vec[i,0]*un0 + n_vec[i,1]*un1 + n_vec[i,2]*un2)
        )

    return tot_def.flatten(), reward, dT




class WheelEnv(gym.Env):

    def __init__(self,
                len_theta=360,
                n_spokes=36, 
                random_spoke_n = 5,
                random_spoke_turns_max = 2,

                render=False, 
                reward_func="percentage", 
                action_space_selection="continous",
                state_space_selection = "rimpoints",



                ):
        
        super().__init__()

        self.len_theta = len_theta
        self.n_spokes = n_spokes
        self.episode_counter = 0
        self.max_tension = 3
        self.global_step_count = 0
        self.action_space_selection = action_space_selection
        self.spoke_turns = np.zeros(self.n_spokes)
        self.reward_func = reward_func

        self.random_spoke_n = random_spoke_n
        self.random_spoke_turns_max = random_spoke_turns_max

        self.state_space_selection = state_space_selection

        self.adjustment_per_turn = 25.4 / 56 / 1000




        self.theta = np.linspace(-np.pi, np.pi, 360)
        self.first_reward = 0
        self.best_reward = 0

        # displacement of the rimpoints 
        if state_space_selection == "rimpoints":
            self.observation_space = gym.spaces.Box(
                low=-50.0, 
                high=50.0, 
                shape=(1080,), 
                dtype=np.float32
            )
        
        if state_space_selection == "spoketensions":
            self.observation_space = gym.spaces.Box(
                low=400.0, 
                high=1200.0, 
                shape=(self.n_spokes,), 
                dtype=np.float32
            )


        # One continuous dimension for "which spoke" (treated as continuous index)
        if self.action_space_selection == "continous":
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, -1.0]),
                high=np.array([float(self.n_spokes - 1), 1.0]),
                dtype=np.float32
            )

        # discrete action selection 
        elif self.action_space_selection == "discrete":
            self.action_space = gym.spaces.Discrete(72)

        #all spokes can be adjusted at once
        elif self.action_space_selection == "all_spokes":
            self.action_space = gym.spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(self.n_spokes,), 
                dtype=np.float32
            )


        ### Wheel Parameters ###
        hub_width = 0.05
        hub_diameter = 0.04

        rim_radius = 0.3
        rim_area = 100e-6
        rim_I_lat = 1500e-12
        rim_I_rad = 3000e-12
        rim_J_tor = 500e-12
        rim_young_mod = 69e9
        rim_shear_mod = 26e9
        rim_I_warp = 0.0

        spokes_crossings = 3
        spokes_diameter = 2.0e-3
        spokes_young_mod = 210e9
        number_modes = 40
        init_tension = 800.

        # Create wheel and rim
        self.wheel = BicycleWheel()
        self.wheel.hub = Hub(width=hub_width, diameter=hub_diameter)
        self.wheel.rim = Rim(
            radius=rim_radius, 
            area=rim_area,
            I_lat=rim_I_lat, 
            I_rad=rim_I_rad, 
            J_tor=rim_J_tor, 
            I_warp=rim_I_warp,
            young_mod=rim_young_mod, 
            shear_mod=rim_shear_mod
        )
        self.wheel.lace_cross(
            n_spokes=n_spokes, 
            n_cross=spokes_crossings, 
            diameter=spokes_diameter,
            young_mod=spokes_young_mod
        )

        # Create a ModeMatrix
        self.mm = ModeMatrix(self.wheel, N=number_modes)

        # Each shape: (len_theta, len(dm))
        self.B_lat = self.mm.B_theta(self.theta, 0)
        self.B_rad = self.mm.B_theta(self.theta, 1)
        self.B_tan = self.mm.B_theta(self.theta, 2)

        # Apply spokes tension
        self.wheel.apply_tension(init_tension)
        self.K = (self.mm.K_rim(tension=True) + 
                  self.mm.K_spk(smeared_spokes=False, tension=True))
        self.F_matrix = self.mm.A_adj()

        self.last_state_norm = 0
        self.best_state_norm = 0
        self._prepare_numba_spoke_arrays()

    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation."""
        super().reset(seed=seed)
        
        self.episode_counter = 0
        self.spoke_turns = np.zeros(self.n_spokes)
        n_random = min(self.random_spoke_n, self.n_spokes)
        random_indices = np.random.choice(self.n_spokes, size=n_random, replace=False)
        self.spoke_turns[random_indices] = np.random.rand(n_random) * self.random_spoke_turns_max - (self.random_spoke_turns_max/2)
        self.tensionchanges = self.spoke_turns * self.adjustment_per_turn
        self.previous_turns = self.spoke_turns.copy()
        
        state, state_norm, _ = self.wheel_calc_new(self.tensionchanges)
        self.last_state_norm = state_norm
        self.first_state_norm = state_norm
        
        # Update best reward if improved
        discard_,self.best_state_norm,discard2_ = self.wheel_calc_new(tensionchanges=((self.spoke_turns % 0.1) * self.adjustment_per_turn))
        #print(reward,'  ',self.best_reward)
        
        info = {"spokes": self.tensionchanges,
                "raw state norm": state_norm,
                "best state norm": self.best_state_norm,
                }
        state = np.clip(state.astype(np.float32), 
                       self.observation_space.low, 
                       self.observation_space.high)
        
        # Gymnasium expects (observation, info) tuple
        return state, info

    def render(self, mode='human'):
        """Optionally implement rendering."""
        pass
    


    def step(self, action):


        if self.action_space_selection == "discrete":
            spoke_index = action // 2
            adjustment = -0.1 if action % 2 == 0 else 0.1

            self.previous_turns = np.copy(self.spoke_turns)
            self.spoke_turns[spoke_index] += adjustment
            self.tensionchanges = self.spoke_turns * self.adjustment_per_turn
        
        elif self.action_space_selection == "continous":
            spoke_index = int(np.clip(np.round(action[0]), 0, self.n_spokes - 1))
            delta = float(np.clip(action[1], -1.0, 1.0))
            self.previous_turns = np.copy(self.spoke_turns)
            self.spoke_turns[spoke_index] += delta
            self.tensionchanges = self.spoke_turns * self.adjustment_per_turn
        
        elif self.action_space_selection == "all_spokes":
                self.spoke_turns += action
                self.tensionchanges = self.spoke_turns * self.adjustment_per_turn



        next_state, state_norm, _ = self.wheel_calc_new(self.tensionchanges)
        wheel_improvement = 100 * (state_norm - self.first_state_norm) / (abs(self.first_state_norm) + 1e-6)
        step_improvement = 100 * (state_norm - self.last_state_norm) / (abs(self.last_state_norm) + 1e-6)
        
        # Compute improvement reward
        if self.reward_func == "raw":
            reward = state_norm
        
        elif self.reward_func == "percentage":
            reward = step_improvement

        
        elif self.reward_func == "normalized":

            if step_improvement > 0:
                reward = 1
            elif step_improvement <= 0:
                reward = -1.0
        
        elif self.reward_func == "spoke":
            if np.all(np.abs(self.previous_turns) >= np.abs(self.spoke_turns)):
                reward = 1
            elif np.all(np.abs(self.previous_turns) <= np.abs(self.spoke_turns)):
                reward = -1


        
        self.last_state_norm = state_norm
        self.episode_counter += 1
        self.global_step_count +=1
        
        # Termination conditions
        truncated = self.episode_counter > 40  # Time limit
        terminated = state_norm >= self.best_state_norm
        
        if terminated:
            reward = 50


        

        info = {"turns": self.spoke_turns,
                "raw state norm": state_norm,
                "improvement": wheel_improvement,
                }
        
        
        return next_state, reward, terminated, truncated, info

    def close(self):
        """Close the environment."""
        pass

    def _prepare_numba_spoke_arrays(self):
        spokes = self.wheel.spokes
        n = len(spokes)

        # Allocate arrays for Numba
        self.n_vec = np.zeros((n, 3), dtype=np.float64)
        self.b_vec = np.zeros((n, 3), dtype=np.float64)
        self.EA = np.zeros(n, dtype=np.float64)
        self.lengths = np.zeros(n, dtype=np.float64)

        # NEW: B_spk[i] = B_theta(theta_spoke_i)
        # Shape is (n_spokes, 4 + 8*n_modes)
        dof = 4 + 8 * self.mm.n_modes
        self.B_spk = np.zeros((n, 4, dof), dtype=np.float64)


        # Also keep track of the spoke's angular index relative to rim θ grid (optional)
        self.spoke_theta_index = np.zeros(n, dtype=np.int64)

        for i, s in enumerate(spokes):
            # Direction vector
            self.n_vec[i] = s.n
            # Vector from rim point to hub eyelet
            self.b_vec[i] = s.b
            # EA stiffness
            self.EA[i] = s.EA
            # Spoke length
            self.lengths[i] = s.length

            # --- Compute B_spk row ---
            theta_i = s.rim_pt[1]            # spoke nipple angle
            B_i = self.mm.B_theta(theta_i)   # shape (4, dof)
            self.B_spk[i, :, :] = B_i     # shape (4, dof)


            # (Optional) nearest rim θ index (still used in your nn state)
            self.spoke_theta_index[i] = np.argmin(np.abs(self.theta - theta_i))



    def wheel_calc_new(self, tensionchanges):

        next_state, reward, tensions = fast_wheel_calc_with_tension(
            self.K,
            self.F_matrix,
            self.B_rad,
            self.B_lat,
            self.B_tan,
            tensionchanges.astype(np.float64),
            self.n_vec,
            self.b_vec,
            self.EA,
            self.lengths,
            self.B_spk,
        )

        # If the state space is spoke tensions, we return that
        if self.state_space_selection == "spoketensions":
            return tensions.astype(np.float32), reward, False

        return next_state.astype(np.float32), reward, False
    
    def wheel_calc_old(self, tensionchanges):
        """Calculate wheel deformation and reward."""
        
        next_state, reward = fast_wheel_calc(
            self.K,
            self.F_matrix,
            self.B_rad,
            self.B_lat,
            self.B_tan,
            tensionchanges.astype(np.float64),
            self.state_space_selection # this is here as a reminder that we might be able to optimize spoke teension calculation in the same manner (njit)
        )

        done = False


        return next_state.astype(np.float32), reward, done

    


import numpy as np
from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix


# --- Initialize a tiny wheel with 1 spoke ---
env = WheelEnv(n_spokes=36, state_space_selection='spoketensions')


# Override some settings for simplicity
env.random_spoke_n = 0  # don't randomize at reset
env.reward_func = "raw"

# Reset environment
state, info = env.reset()

print("Initial spoke turns:", env.spoke_turns)
print("Initial tension changes (should be zero):", env.tensionchanges)

# Let's apply some steps
actions = [0.5, 0.5, -0.5, -0.5]  # adjust in one direction then back

all_tensions = []
all_dT = []

for i, a in enumerate(actions):
    # For continuous action space, action = [spoke_index, delta_turns]
    action = np.array([0, a], dtype=np.float32)
    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep {i+1}:")
    print("  Applied turn:", a)
    print("  Spoke turns:", env.spoke_turns)
    print("  Tension changes dT:", next_state if env.state_space_selection == 'spoketensions' else 'n/a')

    # Track cumulative tension
    current_tension = 800. + next_state
    print("  Current tension (initial + changes):", current_tension)

    all_tensions.append(current_tension)
    all_dT.append(env.tensionchanges[0])

import time 

env = WheelEnv()
start = time.time()
for _ in range(10000):
    spoke_turns = np.zeros(36)
    adjustment_per_turn = 25.4 / 56 / 1000
    n_random = 5
    random_spoke_turns_max = 2
    random_indices = np.random.choice(36, size=n_random, replace=False)
    spoke_turns[random_indices] = np.random.rand(n_random) * random_spoke_turns_max - (random_spoke_turns_max/2)
    tensionchanges = spoke_turns * adjustment_per_turn
    old_tensions = env.wheel_calc_old(tensionchanges)
end = time.time()
print("Old wheel_calc time:", end - start)

# New calc
start = time.time()
for _ in range(10000):
    spoke_turns = np.zeros(36)
    adjustment_per_turn = 25.4 / 56 / 1000
    n_random = 5
    random_spoke_turns_max = 2
    random_indices = np.random.choice(36, size=n_random, replace=False)
    spoke_turns[random_indices] = np.random.rand(n_random) * random_spoke_turns_max - (random_spoke_turns_max/2)
    tensionchanges = spoke_turns * adjustment_per_turn
    new_tensions = env.wheel_calc_new(tensionchanges)
end = time.time()
print("New wheel_calc time:", end - start)