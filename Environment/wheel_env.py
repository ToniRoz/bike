import gymnasium as gym 

from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix

import numpy as np

from numba import njit

"""
Todo:
add new state spaces:
     add code for current (only rimpoints state) (done)
     look up how to calculate the spoketensions (can we jit this as well?) (done)
     add the other two sate space configurations -> we actually dont need to track a tension array since we can go from init (done)
     look through the whole code and remove / edit gpt comments (started)
     name variables clearer for better readability (done)
     add all the options we might want to the env config (number of spokes to turn, max turns, penalty for max)
     things to add:     wheel parameters
                        len theta and n spokes need to be connected to statespace
                        add option for succes reward and tension max (implement to the right units and compare to calc tension)
                        starter tension
     try changing the end goal to depending on total displacement
     try adding fourier state
     get rid of the 800 in tensionstate (and track down where we need to change tracking for it and why tdmpc turns does not work)

     a render option would be nice
"""




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
    tot_def[:, 0] = rad_def * 1000 # with adjustment per turn the units here are in [m] and we convert to [mm]
    tot_def[:, 1] = lat_def * 1000
    tot_def[:, 2] = tan_def * 1000

    # -------------------------
    # tension computation: d = B_theta(θ_spoke) @ dm taken from the original sim and tested for equal output
    # -------------------------
    n_spokes = len(tensionchanges)
    dT = np.empty(n_spokes)

    for i in range(n_spokes):
        # Compute d vector 
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

    return tot_def.flatten(), dT+800 # we add 800 from starter tension (this should be an env variable)




class WheelEnv(gym.Env):

    def __init__(self,
                 
                # state space 
                len_theta=100,
                n_spokes=36,

                random_spoke_n = 5,
                random_spoke_turns_max = 2,

                render=False,

                #reward function 
                max_tension_penalty = False,
                max_tension_threshold = 0,
                goal_condition ="modulo", 
                reward_func="percentage", 
                action_space_selection="continous",
                state_space_selection = "rimpoints",


                # wheel sim parameters:
                hub_width = 0.05,
                hub_diameter = 0.04,

                rim_radius = 0.3,
                rim_area = 100e-6,
                rim_I_lat = 1500e-12,
                rim_I_rad = 3000e-12,
                rim_J_tor = 500e-12,
                rim_young_mod = 69e9,
                rim_shear_mod = 26e9,
                rim_I_warp = 0.0,

                spokes_crossings = 3,
                spokes_diameter = 2.0e-3,
                spokes_young_mod = 210e9,
                number_modes = 40,
                init_tension = 800.,



                ):
        
        super().__init__()

        self.n_spokes = n_spokes
        self.episode_counter = 0
        self.max_tension = 3 # here we should express this in tension instead of turns and relate it to the calculated tension
        self.global_step_count = 0
        self.action_space_selection = action_space_selection
        self.spoke_turns = np.zeros(self.n_spokes)
        self.reward_func = reward_func

        self.random_spoke_n = random_spoke_n
        self.random_spoke_turns_max = random_spoke_turns_max

        self.state_space_selection = state_space_selection

        self.adjustment_per_turn = 25.4 / 56 / 1000
        self.reward_func = reward_func
        



        # in the following we need to make the state space depending on nspokes and make another input for the number of points
        self.theta = np.linspace(-np.pi, np.pi, len_theta)
        self.first_reward = 0
        self.best_reward = 0

        # displacement of the rimpoints 
        if state_space_selection == "rimpoints":
            self.observation_space = gym.spaces.Box(
                low=-50.0, 
                high=50.0, 
                shape=(len_theta*3,), 
                dtype=np.float32
            )
        
        if state_space_selection == "rimandspokes":
            self.observation_space = gym.spaces.Box(
                low=-50.0, 
                high=1200.0, 
                shape=(len_theta*3 + self.n_spokes,), 
                dtype=np.float32
            )

        if state_space_selection == "spoketensions":
            self.observation_space = gym.spaces.Box(
                low=400.0, 
                high=1200.0, 
                shape=(self.n_spokes ,), 
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
        self.first_tensions = np.zeros(n_spokes)
        self.tensions = np.zeros(n_spokes)
        self._prepare_numba_spoke_arrays()

    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation."""
        super().reset(seed=seed)
        
        self.episode_counter = 0

        # randomize the spoke-turns for randomly selected number of spokes 
        self.spoke_turns = np.zeros(self.n_spokes)
        n_random = min(self.random_spoke_n, self.n_spokes)
        random_indices = np.random.choice(self.n_spokes, size=n_random, replace=False)
        self.spoke_turns[random_indices] = np.random.rand(n_random) * self.random_spoke_turns_max - (self.random_spoke_turns_max/2)

        self.tensionchanges = self.spoke_turns * self.adjustment_per_turn
        self.previous_turns = self.spoke_turns.copy()
        
        # calculate wheel displacement and spoketensions
        wheel_displacement, tensions = self.wheel_calc(self.tensionchanges)
        state_norm = np.linalg.norm(wheel_displacement)
        self.tensions = tensions
        self.first_tensions = self.tensions
        self.last_state_norm = state_norm
        self.first_state_norm = state_norm
        
        # calculate an estimation of a good endstate by taking the residuals of turns when minimized by discrete adjsutment-step-size
        best_displacement, best_tensions = self.wheel_calc(tensionchanges=((self.spoke_turns % 0.1) * self.adjustment_per_turn))
        self.best_state_norm = np.linalg.norm(best_displacement)
        
        info = {"spoke turns": self.spoke_turns,
                "raw state norm": state_norm,
                "best state norm": self.best_state_norm,
                "spoke tensions": self.tensions,
                }

        

        if self.state_space_selection == "spoketensions":
            return tensions.astype(np.float32), info
        
        if self.state_space_selection == "rimandspokes":
            combined_state = np.concatenate([wheel_displacement, tensions])
            return combined_state.astype(np.float32), info
        
        if self.state_space_selection == "rimpoints":
            return wheel_displacement.astype(np.float32), info
        


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



        wheel_displacement, tensions= self.wheel_calc(self.tensionchanges)
        state_norm = np.linalg.norm(wheel_displacement)
        wheel_improvement = 100 * ( self.first_state_norm - state_norm ) / (abs(self.first_state_norm) + 1e-6)
        step_improvement = 100 * (self.first_state_norm - state_norm) / (abs(self.last_state_norm) + 1e-6)
        
        # Compute improvement reward
        if self.reward_func == "raw":
            reward = -state_norm
        
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
        terminated = state_norm <= self.best_state_norm # 'best' state reached
        
        if terminated:
            reward = 50


        

        info = {"spoke turns": self.spoke_turns,
                "raw state norm": state_norm,
                "improvement": wheel_improvement,
                "spoke tensions": tensions
                }
        

        if self.state_space_selection == "spoketensions":
            return tensions.astype(np.float32), reward, terminated, truncated, info
        
        if self.state_space_selection == "rimandspokes":
            combined_state = np.concatenate([wheel_displacement, (tensions-800)/100])
            return combined_state.astype(np.float32), reward, terminated, truncated, info
        
        if self.state_space_selection == "rimpoints":
            return wheel_displacement.astype(np.float32), reward, terminated, truncated, info
    

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



    def wheel_calc(self, tensionchanges):

        wheel_displacement,  tensions = fast_wheel_calc_with_tension(
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
        return wheel_displacement, tensions




# should add a real test here
"""
env = WheelEnv(reward_func="percentage", action_space_selection="discrete")
print(env.reward_func)
print("test")

state, info = env.reset()
first_turns = np.sum(abs(info['spoke turns']))
print("start turns:", first_turns)

state, reward, truncated, terminated, info = env.step(1)
print("reward:", reward)
print(info)

state, reward, truncated, terminated, info = env.step(1)
print("reward:", reward)
print(info)

state, reward, truncated, terminated, info = env.step(1)
print("reward:", reward)
print(info)
current_turns = np.sum(abs(info['spoke turns']))
print("current turns:", current_turns)
turn_change = 100 * (first_turns - current_turns) / max(abs(first_turns), 1e-15)
print("turn change:", turn_change)

state, reward, truncated, terminated, info = env.step(-1)
print("reward:", reward)
print(info)

state, reward, truncated, terminated, info = env.step(-1)
print("reward:", reward)
print(info)

state, reward, truncated, terminated, info = env.step(-1)
print("reward:", reward)
print(info)
"""
#!/usr/bin/env python3
"""
Test script to analyze WheelEnv state distributions
Collects statistics on wheel displacement and spoke tensions
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Assuming your environment is importable
# from your_module import WheelEnv


def collect_statistics(env, n_episodes=100, steps_per_episode=50):
    """
    Collect statistics by taking random actions in the environment.
    
    Returns:
        stats: Dictionary containing all collected statistics
    """
    
    stats = {
        # Reset statistics
        'reset_displacement': [],
        'reset_displacement_flat': [],
        'reset_tensions': [],
        'reset_state_norm': [],
        'reset_tension_mean': [],
        'reset_tension_std': [],
        
        # Step statistics
        'step_displacement': [],
        'step_displacement_flat': [],
        'step_tensions': [],
        'step_state_norm': [],
        'step_tension_mean': [],
        'step_tension_std': [],
        'step_tension_min': [],
        'step_tension_max': [],
        'step_tension_range': [],
        
        # Per-spoke statistics
        'all_spoke_tensions': [],
        'all_displacements': [],
        
        # Rewards
        'rewards': [],
        'episode_returns': [],
    }
    
    print(f"Collecting statistics from {n_episodes} episodes...")
    print(f"Steps per episode: {steps_per_episode}")
    print("="*60)
    
    for episode in range(n_episodes):
        # Reset environment
        state, info = env.reset()
        
        # Extract displacement and tensions from reset
        if env.state_space_selection == "rimpoints":
            displacement = state
            tensions = info['spoke tensions']
        elif env.state_space_selection == "rimandspokes":
            displacement = state[:1080]
            tensions = state[1080:]
        elif env.state_space_selection == "spoketensions":
            displacement = None
            tensions = state
        
        # Store reset statistics
        if displacement is not None:
            stats['reset_displacement'].append(displacement)
            stats['reset_displacement_flat'].extend(displacement.flatten())
            stats['reset_state_norm'].append(info['raw state norm'])
        
        stats['reset_tensions'].append(tensions)
        stats['reset_tension_mean'].append(tensions.mean())
        stats['reset_tension_std'].append(tensions.std())
        stats['all_spoke_tensions'].append(tensions)
        
        episode_return = 0
        
        # Take random steps
        for step in range(steps_per_episode):
            # Random action
            action = env.action_space.sample()
            
            # Step
            state, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            
            # Extract displacement and tensions from step
            if env.state_space_selection == "rimpoints":
                displacement = state
                tensions = info['spoke tensions']
            elif env.state_space_selection == "rimandspokes":
                displacement = state[:1080]
                tensions = state[1080:]
            elif env.state_space_selection == "spoketensions":
                displacement = None
                tensions = state
            
            # Store step statistics
            if displacement is not None:
                stats['step_displacement'].append(displacement)
                stats['step_displacement_flat'].extend(displacement.flatten())
                stats['step_state_norm'].append(info['raw state norm'])
                stats['all_displacements'].append(displacement)
            
            stats['step_tensions'].append(tensions)
            stats['step_tension_mean'].append(tensions.mean())
            stats['step_tension_std'].append(tensions.std())
            stats['step_tension_min'].append(tensions.min())
            stats['step_tension_max'].append(tensions.max())
            stats['step_tension_range'].append(tensions.max() - tensions.min())
            stats['all_spoke_tensions'].append(tensions)
            
            stats['rewards'].append(reward)
            
            if terminated or truncated:
                break
        
        stats['episode_returns'].append(episode_return)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} completed")
    
    return stats


def analyze_statistics(stats):
    """
    Analyze and print statistics in a clear format.
    """
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*60)
    
    # Convert lists to arrays for easier analysis
    reset_tensions = np.array(stats['reset_tensions'])
    step_tensions = np.array(stats['step_tensions'])
    all_tensions = np.array(stats['all_spoke_tensions'])
    
    print("\n" + "-"*60)
    print("SPOKE TENSIONS ANALYSIS")
    print("-"*60)
    
    print("\n1. AFTER RESET (Initial State):")
    print(f"   Mean tension:    {reset_tensions.mean():.2f} N")
    print(f"   Std tension:     {reset_tensions.std():.2f} N")
    print(f"   Min tension:     {reset_tensions.min():.2f} N")
    print(f"   Max tension:     {reset_tensions.max():.2f} N")
    print(f"   Range:           {reset_tensions.max() - reset_tensions.min():.2f} N")
    print(f"   Median:          {np.median(reset_tensions):.2f} N")
    print(f"   25th percentile: {np.percentile(reset_tensions, 25):.2f} N")
    print(f"   75th percentile: {np.percentile(reset_tensions, 75):.2f} N")
    
    print("\n2. DURING EPISODES (After Random Actions):")
    print(f"   Mean tension:    {step_tensions.mean():.2f} N")
    print(f"   Std tension:     {step_tensions.std():.2f} N")
    print(f"   Min tension:     {step_tensions.min():.2f} N")
    print(f"   Max tension:     {step_tensions.max():.2f} N")
    print(f"   Range:           {step_tensions.max() - step_tensions.min():.2f} N")
    print(f"   Median:          {np.median(step_tensions):.2f} N")
    print(f"   25th percentile: {np.percentile(step_tensions, 25):.2f} N")
    print(f"   75th percentile: {np.percentile(step_tensions, 75):.2f} N")
    
    print("\n3. TENSION STATISTICS PER SPOKE (Across All States):")
    tension_per_spoke = all_tensions.T  # Shape: (n_spokes, n_samples)
    print(f"   Number of spokes: {tension_per_spoke.shape[0]}")
    print(f"   Samples per spoke: {tension_per_spoke.shape[1]}")
    for i in range(min(5, tension_per_spoke.shape[0])):
        print(f"   Spoke {i}: mean={tension_per_spoke[i].mean():.2f}N, "
              f"std={tension_per_spoke[i].std():.2f}N, "
              f"range=[{tension_per_spoke[i].min():.2f}, {tension_per_spoke[i].max():.2f}]N")
    if tension_per_spoke.shape[0] > 5:
        print(f"   ... (showing first 5 of {tension_per_spoke.shape[0]} spokes)")
    
    # Wheel displacement analysis (if available)
    if stats['step_displacement_flat']:
        print("\n" + "-"*60)
        print("WHEEL DISPLACEMENT ANALYSIS")
        print("-"*60)
        
        reset_disp = np.array(stats['reset_displacement_flat'])
        step_disp = np.array(stats['step_displacement_flat'])
        
        print("\n1. AFTER RESET (Initial State):")
        print(f"   Mean displacement:    {reset_disp.mean():.6f} m")
        print(f"   Std displacement:     {reset_disp.std():.6f} m")
        print(f"   Min displacement:     {reset_disp.min():.6f} m")
        print(f"   Max displacement:     {reset_disp.max():.6f} m")
        print(f"   Abs mean:             {np.abs(reset_disp).mean():.6f} m")
        print(f"   Range:                {reset_disp.max() - reset_disp.min():.6f} m")
        
        print("\n2. DURING EPISODES (After Random Actions):")
        print(f"   Mean displacement:    {step_disp.mean():.6f} m")
        print(f"   Std displacement:     {step_disp.std():.6f} m")
        print(f"   Min displacement:     {step_disp.min():.6f} m")
        print(f"   Max displacement:     {step_disp.max():.6f} m")
        print(f"   Abs mean:             {np.abs(step_disp).mean():.6f} m")
        print(f"   Range:                {step_disp.max() - step_disp.min():.6f} m")
        
        # State norm analysis
        reset_norms = np.array(stats['reset_state_norm'])
        step_norms = np.array(stats['step_state_norm'])
        
        print("\n3. STATE NORM (L2 norm of displacement vector):")
        print(f"   Reset - Mean: {reset_norms.mean():.6f}, Std: {reset_norms.std():.6f}")
        print(f"   Reset - Range: [{reset_norms.min():.6f}, {reset_norms.max():.6f}]")
        print(f"   Steps - Mean: {step_norms.mean():.6f}, Std: {step_norms.std():.6f}")
        print(f"   Steps - Range: [{step_norms.min():.6f}, {step_norms.max():.6f}]")
    
    # Reward analysis
    print("\n" + "-"*60)
    print("REWARD ANALYSIS")
    print("-"*60)
    rewards = np.array(stats['rewards'])
    returns = np.array(stats['episode_returns'])
    
    print(f"\nStep rewards:")
    print(f"   Mean:   {rewards.mean():.2f}")
    print(f"   Std:    {rewards.std():.2f}")
    print(f"   Min:    {rewards.min():.2f}")
    print(f"   Max:    {rewards.max():.2f}")
    print(f"   Median: {np.median(rewards):.2f}")
    
    print(f"\nEpisode returns:")
    print(f"   Mean:   {returns.mean():.2f}")
    print(f"   Std:    {returns.std():.2f}")
    print(f"   Min:    {returns.min():.2f}")
    print(f"   Max:    {returns.max():.2f}")
    print(f"   Median: {np.median(returns):.2f}")
    
    return {
        'reset_tensions': reset_tensions,
        'step_tensions': step_tensions,
        'all_tensions': all_tensions,
        'reset_disp': np.array(stats['reset_displacement_flat']) if stats['reset_displacement_flat'] else None,
        'step_disp': np.array(stats['step_displacement_flat']) if stats['step_displacement_flat'] else None,
        'reset_norms': np.array(stats['reset_state_norm']) if stats['reset_state_norm'] else None,
        'step_norms': np.array(stats['step_state_norm']) if stats['step_state_norm'] else None,
    }


def plot_distributions(arrays_dict, save_path='wheel_env_distributions.png'):
    """
    Plot distributions of key variables.
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('WheelEnv State Distributions', fontsize=16, fontweight='bold')
    
    # Spoke tensions after reset
    ax = axes[0, 0]
    if arrays_dict['reset_tensions'] is not None:
        ax.hist(arrays_dict['reset_tensions'].flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(arrays_dict['reset_tensions'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Spoke Tension (N)')
        ax.set_ylabel('Frequency')
        ax.set_title('Tensions After Reset')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Spoke tensions during episodes
    ax = axes[0, 1]
    if arrays_dict['step_tensions'] is not None:
        ax.hist(arrays_dict['step_tensions'].flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(arrays_dict['step_tensions'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Spoke Tension (N)')
        ax.set_ylabel('Frequency')
        ax.set_title('Tensions During Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Comparison: Reset vs Step tensions
    ax = axes[0, 2]
    if arrays_dict['reset_tensions'] is not None and arrays_dict['step_tensions'] is not None:
        ax.hist(arrays_dict['reset_tensions'].flatten(), bins=50, alpha=0.5, 
                color='blue', label='Reset', edgecolor='black')
        ax.hist(arrays_dict['step_tensions'].flatten(), bins=50, alpha=0.5, 
                color='green', label='Steps', edgecolor='black')
        ax.set_xlabel('Spoke Tension (N)')
        ax.set_ylabel('Frequency')
        ax.set_title('Tension Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Wheel displacement after reset
    ax = axes[1, 0]
    if arrays_dict['reset_disp'] is not None:
        ax.hist(arrays_dict['reset_disp'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.set_xlabel('Displacement (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Displacement After Reset')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Wheel displacement during episodes
    ax = axes[1, 1]
    if arrays_dict['step_disp'] is not None:
        ax.hist(arrays_dict['step_disp'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.set_xlabel('Displacement (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Displacement During Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # State norms
    ax = axes[1, 2]
    if arrays_dict['reset_norms'] is not None and arrays_dict['step_norms'] is not None:
        ax.hist(arrays_dict['reset_norms'], bins=50, alpha=0.5, 
                color='blue', label='Reset', edgecolor='black')
        ax.hist(arrays_dict['step_norms'], bins=50, alpha=0.5, 
                color='green', label='Steps', edgecolor='black')
        ax.set_xlabel('State Norm (L2)')
        ax.set_ylabel('Frequency')
        ax.set_title('State Norm Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    plt.show()


def suggest_normalization(arrays_dict):
    """
    Suggest normalization strategies based on the collected statistics.
    """
    
    print("\n" + "="*60)
    print("NORMALIZATION RECOMMENDATIONS")
    print("="*60)
    
    # Tension normalization
    if arrays_dict['step_tensions'] is not None:
        tensions = arrays_dict['step_tensions'].flatten()
        t_mean = tensions.mean()
        t_std = tensions.std()
        t_min = tensions.min()
        t_max = tensions.max()
        
        print("\n1. SPOKE TENSIONS:")
        print(f"   Range: [{t_min:.2f}, {t_max:.2f}] N")
        print(f"   Mean ± Std: {t_mean:.2f} ± {t_std:.2f} N")
        
        print("\n   Recommended normalization options:")
        print(f"\n   Option A: Z-score normalization")
        print(f"      normalized = (tension - {t_mean:.2f}) / {t_std:.2f}")
        print(f"      Result range: approximately [-3, +3]")
        
        print(f"\n   Option B: Min-Max to [-1, 1]")
        print(f"      normalized = 2 * (tension - {t_min:.2f}) / {t_max - t_min:.2f} - 1")
        print(f"      Result range: exactly [-1, +1]")
        
        print(f"\n   Option C: Deviation from nominal (assuming 800N nominal)")
        nominal = 800
        print(f"      normalized = (tension - {nominal}) / {t_std:.2f}")
        print(f"      Centers around 0 for nominal tension")
        
        print(f"\n   Option D: Percentage deviation from nominal")
        print(f"      normalized = (tension - {nominal}) / {nominal}")
        print(f"      Result is fractional change from nominal")
    
    # Displacement normalization
    if arrays_dict['step_disp'] is not None:
        disp = arrays_dict['step_disp']
        d_mean = disp.mean()
        d_std = disp.std()
        d_min = disp.min()
        d_max = disp.max()
        d_abs_mean = np.abs(disp).mean()
        
        print("\n2. WHEEL DISPLACEMENT:")
        print(f"   Range: [{d_min:.6f}, {d_max:.6f}] m")
        print(f"   Mean ± Std: {d_mean:.6f} ± {d_std:.6f} m")
        print(f"   Abs mean: {d_abs_mean:.6f} m")
        
        print("\n   Recommended normalization options:")
        print(f"\n   Option A: Z-score normalization")
        print(f"      normalized = (displacement - {d_mean:.6f}) / {d_std:.6f}")
        print(f"      Result range: approximately [-3, +3]")
        
        print(f"\n   Option B: Division by std only (already ~centered at 0)")
        print(f"      normalized = displacement / {d_std:.6f}")
        print(f"      Preserves zero-centering")
        
        print(f"\n   Option C: Scale by typical magnitude")
        print(f"      normalized = displacement / {d_abs_mean:.6f}")
        print(f"      Based on typical absolute displacement")
    
    # Combined state
    if arrays_dict['step_tensions'] is not None and arrays_dict['step_disp'] is not None:
        print("\n3. COMBINED STATE (displacement + tensions):")
        print("\n   ⚠️  WARNING: Very different scales!")
        print(f"      Displacement std: {arrays_dict['step_disp'].std():.6f} m")
        print(f"      Tension std:      {arrays_dict['step_tensions'].std():.2f} N")
        print(f"      Ratio: {arrays_dict['step_tensions'].std() / arrays_dict['step_disp'].std():.1e}")
        
        print("\n   Recommended approach:")
        print("      1. Normalize displacement: disp / disp_std")
        print("      2. Normalize tensions: (tension - 800) / tension_std")
        print("      3. Concatenate normalized values")
        print("\n   This ensures both contribute equally to gradients.")
    
    print("\n" + "="*60)
    print("CODE SNIPPETS FOR NORMALIZATION")
    print("="*60)
    
    if arrays_dict['step_tensions'] is not None:
        t_mean = arrays_dict['step_tensions'].mean()
        t_std = arrays_dict['step_tensions'].std()
        print("\n# Tension normalization (add to your environment):")
        print(f"tensions_normalized = (tensions - {t_mean:.2f}) / {t_std:.2f}")
        print("# Or deviation from nominal:")
        print(f"tensions_normalized = (tensions - 800.0) / {t_std:.2f}")
    
    if arrays_dict['step_disp'] is not None:
        d_std = arrays_dict['step_disp'].std()
        print("\n# Displacement normalization:")
        print(f"displacement_normalized = displacement / {d_std:.6f}")
    
    if arrays_dict['step_tensions'] is not None and arrays_dict['step_disp'] is not None:
        print("\n# Combined state normalization:")
        print(f"""
displacement_norm = displacement / {arrays_dict['step_disp'].std():.6f}
tensions_norm = (tensions - 800.0) / {arrays_dict['step_tensions'].std():.2f}
state = np.concatenate([displacement_norm, tensions_norm])
""")


def main():
    """
    Main function to run the test.
    """
    
    # Import your environment here
    # Replace with your actual import
    try:
        from wheel_env import WheelEnv  # Adjust this import!
    except ImportError:
        print("ERROR: Could not import WheelEnv")
        print("Please adjust the import statement in this script")
        return
    
    print("\n" + "="*60)
    print("WHEELENV STATE STATISTICS TEST")
    print("="*60)
    
    # Test different state space configurations
    state_configs = [
        "rimpoints",
        "rimandspokes",
        # "spoketensions",  # Uncomment if you want to test this too
    ]
    
    for state_space in state_configs:
        print(f"\n\n{'='*60}")
        print(f"TESTING: state_space_selection = '{state_space}'")
        print(f"{'='*60}\n")
        
        # Create environment
        env = WheelEnv(
            state_space_selection=state_space,
            action_space_selection="discrete",
            random_spoke_n=5,
            random_spoke_turns_max=2,
        )
        
        # Collect statistics
        stats = collect_statistics(env, n_episodes=100, steps_per_episode=50)
        
        # Analyze
        arrays_dict = analyze_statistics(stats)
        
        # Plot
        plot_name = f'wheel_env_distributions_{state_space}.png'
        plot_distributions(arrays_dict, save_path=plot_name)
        
        # Suggestions
        suggest_normalization(arrays_dict)
        
        env.close()
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()