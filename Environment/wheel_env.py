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
     things to add:     wheel parameters (done)
                        len theta and n spokes need to be connected to statespace (done)
                        add option for success reward and tension max (implement to the right units and compare to calc tension)
                        starter tension
     try adding fourier state (done)
     get rid of the 800 in tensionstate (and track down where we need to change tracking for it and why tdmpc turns does not work)(done)
     last to finish:
        normalize raw by statesize or use integral formulation
        add penalty (most sources speak of about 100-120 kgf max so i guess we should stop at 1000N)(done but needs adjustment of values)
        add different stoping condition (where spokes are within a certain tension of each other and total and displacment is under threshold)(sorta done)
        add option for tangential displacement include (done)
        add absolute logging
        take a last look at reward functions and document
     
     make it speak when init
     a render option would be nice
"""

def compute_fourier_features(tot_def, n_harmonics=10):
    """
    Compute Fourier coefficients for the rim displacement.
    
    Args:
        tot_def: shape (npts, 2) - [radial, lateral] in mm
        n_harmonics: number of harmonics to keep (excluding DC)
    
    Returns:
        features: array of shape (2 + 4*n_harmonics,)
    """
    npts = tot_def.shape[0]
    
    # Extract components
    rad = tot_def[:, 0]
    lat = tot_def[:, 1]
    
    # Since we only have radial (no tangential), treat radial as the in-plane component
    rad_fft = np.fft.fft(rad) / npts
    lat_fft = np.fft.fft(lat) / npts
    
    # Build feature vector
    features = []
    
    # DC components (k=0)
    features.append(np.real(rad_fft[0]))  # mean radial
    features.append(np.real(lat_fft[0]))  # mean lateral
    
    # Harmonics 1 to n_harmonics
    for k in range(1, n_harmonics + 1):
        # Radial (in-plane)
        features.append(np.abs(rad_fft[k]) * 2)      # magnitude
        features.append(np.angle(rad_fft[k]))        # phase
        
        # Lateral (out-of-plane)
        features.append(np.abs(lat_fft[k]) * 2)      # magnitude
        features.append(np.angle(lat_fft[k]))        # phase
    
    return np.array(features, dtype=np.float32)


@njit
def fast_wheel_calc_with_tension(
    K, F_matrix,
    B_rad, B_lat, B_tan,
    tensionchanges,
    n_vec, b_vec, EA, lengths,
    B_spk,
    include_tan_state
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
    if include_tan_state:
        tot_def = np.empty((npts, 3))
        tot_def[:, 0] = rad_def * 1000 # with adjustment per turn the units here are in [m] and we convert to [mm]
        tot_def[:, 1] = lat_def * 1000
        tot_def[:, 2] = tan_def * 1000
    else:
        tot_def = np.empty((npts, 2))
        tot_def[:, 0] = rad_def * 1000 # with adjustment per turn the units here are in [m] and we convert to [mm]
        tot_def[:, 1] = lat_def * 1000


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

    return tot_def.flatten(), dT 




class WheelEnv(gym.Env):

    def __init__(self,
                 
                # state space 
                len_theta=360,
                n_spokes=36,

                random_spoke_n = 5,
                random_spoke_turns_max = 2,

                render=False,

                #reward function 
                max_tension_penalty = False,
                max_tension_threshold = 1000,
                include_tan_displacement = False,
                goal_condition ="modulo", 
                reward_func="percentage", 
                action_space_selection="continous",
                state_space_selection = "rimpoints",
                n_harmonics = 20,


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
        self.include_tan_displacement = include_tan_displacement

        self.max_tension_threshold = max_tension_threshold
        self.max_tension_penalty = max_tension_penalty

        self.goal_condition = goal_condition 
        self.random_spoke_n = random_spoke_n
        self.random_spoke_turns_max = random_spoke_turns_max

        self.state_space_selection = state_space_selection

        self.adjustment_per_turn = 25.4 / 56 / 1000
        self.reward_func = reward_func

        if self.include_tan_displacement:
            stacksize = 3
        else:
            stacksize = 2
        



        # in the following we need to make the state space depending on nspokes and make another input for the number of points
        self.theta = np.linspace(-np.pi, np.pi, len_theta)
        self.first_reward = 0
        self.best_reward = 0

        # displacement of the rimpoints 
        if state_space_selection == "rimpoints":
            self.observation_space = gym.spaces.Box(
                low=-50.0, 
                high=50.0, 
                shape=(len_theta*stacksize,), 
                dtype=np.float32
            )
        
        if state_space_selection == "rimandspokes":
            self.observation_space = gym.spaces.Box(
                low=-50.0, 
                high=1200.0, 
                shape=(len_theta*stacksize + self.n_spokes,), 
                dtype=np.float32
            )

        if state_space_selection == "spoketensions":
            self.observation_space = gym.spaces.Box(
                low=400.0, 
                high=1200.0, 
                shape=(self.n_spokes ,), 
                dtype=np.float32
            )
        
        self.n_harmonics = n_harmonics
    
        # Add new state space option
        if state_space_selection == "fourier":
            # 3 DC + 4*n_harmonics features
            n_features = 2 + 4 * n_harmonics
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(n_features,), 
                dtype=np.float32
            )
        
        if state_space_selection == "fourier_and_spokes":
            n_features = 2 + 4 * n_harmonics + self.n_spokes
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=1200.0, 
                shape=(n_features,), 
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
        if self.state_space_selection=="fourier" or self.state_space_selection=="fourier_and_spokes":
            wheel_displacement, tensions, fourier_coeffs = self.wheel_calc(self.tensionchanges,True)
        else:
            wheel_displacement, tensions = self.wheel_calc(self.tensionchanges,False)
        state_norm = np.linalg.norm(wheel_displacement)
        self.tensions = tensions
        self.first_tensions = self.tensions
        self.last_state_norm = state_norm
        self.first_state_norm = state_norm
        
        # calculate an estimation of a good endstate by taking the residuals of turns when minimized by discrete adjsutment-step-size
        best_displacement, best_tensions = self.wheel_calc(tensionchanges=((self.spoke_turns % 0.1) * self.adjustment_per_turn),return_fourier=False)
        self.best_state_norm = np.linalg.norm(best_displacement)
        
        info = {"spoke turns": self.spoke_turns,
                "raw state norm": state_norm,
                "best state norm": self.best_state_norm,
                "spoke tensions": self.tensions,
                }

        

        if self.state_space_selection == "spoketensions":
            return tensions.astype(np.float32), info
        
        if self.state_space_selection == "rimandspokes":
            combined_state = np.concatenate([wheel_displacement, tensions/100]) # we might not need to downscale this here but for now it works
            return combined_state.astype(np.float32), info
        
        if self.state_space_selection == "rimpoints":
            return wheel_displacement.astype(np.float32), info
        
        if self.state_space_selection == "fourier":
            return fourier_coeffs, info
    
        if self.state_space_selection == "fourier_and_spokes":

            combined = np.concatenate([fourier_coeffs, tensions/100])
            return combined.astype(np.float32), info
        


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


        if self.state_space_selection=="fourier" or self.state_space_selection=="fourier_and_spokes":
            wheel_displacement, tensions, fourier_coeffs = self.wheel_calc(self.tensionchanges,True)
        else:
            wheel_displacement, tensions= self.wheel_calc(self.tensionchanges,False)
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
        
        elif self.reward_func == "spokes":
            if np.all(np.abs(self.previous_turns) >= np.abs(self.spoke_turns)):
                reward = 1
            elif np.all(np.abs(self.previous_turns) <= np.abs(self.spoke_turns)):
                reward = -1

        
        
        self.last_state_norm = state_norm
        self.episode_counter += 1
        self.global_step_count +=1
        
        if self.max_tension_penalty:
            if np.any(self.tensionchanges + 800) > self.max_tension_threshold: #implement starter tension as variable
                reward = reward - 10 

        # Termination conditions
        truncated = self.episode_counter > 40  # Time limit
        if self.goal_condition=="modulo":
            terminated = state_norm <= self.best_state_norm # 'best' state reached

        else: # absolute displacement of maximum 0.2 mm
            if np.all(wheel_displacement) < 0.2 and np.all(abs(self.tensionchanges)) < 50:
                terminated = True
        
        if terminated:
            reward = 50 # need to take a look at this


        

        info = {"spoke turns": self.spoke_turns,
                "raw state norm": state_norm,
                "improvement": wheel_improvement,
                "spoke tensions": tensions
                }
        

        if self.state_space_selection == "spoketensions":
            return tensions.astype(np.float32), reward, terminated, truncated, info
        
        if self.state_space_selection == "rimandspokes":
            combined_state = np.concatenate([wheel_displacement,tensions/100])
            return combined_state.astype(np.float32), reward, terminated, truncated, info
        
        if self.state_space_selection == "rimpoints":
            return wheel_displacement.astype(np.float32), reward, terminated, truncated, info
        
        if self.state_space_selection == "fourier":
            return fourier_coeffs,reward, terminated, truncated, info
    
        if self.state_space_selection == "fourier_and_spokes":

            combined = np.concatenate([fourier_coeffs, tensions/100])
            return combined.astype(np.float32),reward, terminated, truncated, info
    

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



    def wheel_calc(self, tensionchanges, return_fourier=False):
        """
        Calculate wheel displacement and tensions.
        
        Args:
            tensionchanges: spoke tension adjustments
            return_fourier: if True, also return Fourier features
        """
        wheel_displacement, tensions = fast_wheel_calc_with_tension(
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
            self.include_tan_displacement
        )
        
        if return_fourier:
            if self.include_tan_displacement:   
                tot_def = wheel_displacement.reshape(-1, 3)
                fourier_features = compute_fourier_features(tot_def[:, :2], n_harmonics=self.n_harmonics)
                return wheel_displacement, tensions, fourier_features

            else:
                tot_def = wheel_displacement.reshape(-1, 2)
                fourier_features = compute_fourier_features(tot_def, n_harmonics=self.n_harmonics)
                return wheel_displacement, tensions, fourier_features
        
        return wheel_displacement, tensions


"""
import numpy as np
import matplotlib.pyplot as plt

def reconstruct_from_fourier(fourier_features, npts, n_harmonics):

    # Extract DC components
    dc_rad = fourier_features[0]
    dc_lat = fourier_features[1]
    
    # Initialize reconstruction
    theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
    rad_recon = np.ones(npts) * dc_rad
    lat_recon = np.ones(npts) * dc_lat
    
    # Add harmonics
    for k in range(1, n_harmonics + 1):
        idx_base = 2 + 4*(k-1)
        
        # Radial
        mag_rad = fourier_features[idx_base]
        phase_rad = fourier_features[idx_base + 1]
        
        # Lateral
        mag_lat = fourier_features[idx_base + 2]
        phase_lat = fourier_features[idx_base + 3]
        
        # Reconstruct
        rad_recon += mag_rad * np.cos(k * theta + phase_rad)
        lat_recon += mag_lat * np.cos(k * theta + phase_lat)
    
    return np.column_stack([rad_recon, lat_recon])


def test_reconstruction_quality(env, n_harmonics_list=[2, 4, 8, 12, 16, 20]):

    print("\nTest: Reconstruction quality vs number of harmonics")
    print("="*60)
    
    # Reset environment and get initial state
    state, info = env.reset()
    
    # Get the full displacement
    tensionchanges = env.spoke_turns * env.adjustment_per_turn
    wheel_disp_full, tensions = env.wheel_calc(tensionchanges, return_fourier=False)
    
    # Reshape to (npts, 2)
    npts = len(wheel_disp_full) // 2
    original = wheel_disp_full.reshape(-1, 2)
    
    print(f"Original displacement shape: {original.shape}")
    print(f"Original norm: {np.linalg.norm(original):.6f}")
    
    # Test different numbers of harmonics
    results = []
    
    for n_harm in n_harmonics_list:
        # Compute Fourier features
        fourier_features = compute_fourier_features(original, n_harmonics=n_harm)
        
        # Reconstruct
        reconstructed = reconstruct_from_fourier(fourier_features, npts, n_harm)
        
        # Compute error metrics
        error = original - reconstructed
        rmse = np.sqrt(np.mean(error**2))
        max_error = np.max(np.abs(error))
        relative_error = rmse / (np.linalg.norm(original) + 1e-10)
        
        results.append({
            'n_harmonics': n_harm,
            'n_features': len(fourier_features),
            'rmse': rmse,
            'max_error': max_error,
            'relative_error': relative_error * 100
        })
        
        print(f"\nn_harmonics={n_harm} ({len(fourier_features)} features):")
        print(f"  RMSE: {rmse:.6f} mm")
        print(f"  Max error: {max_error:.6f} mm")
        print(f"  Relative error: {relative_error*100:.2f}%")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Error vs harmonics
    ax = axes[0, 0]
    n_harms = [r['n_harmonics'] for r in results]
    rmses = [r['rmse'] for r in results]
    ax.semilogy(n_harms, rmses, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Harmonics')
    ax.set_ylabel('RMSE [mm]')
    ax.set_title('Reconstruction Error vs Harmonics')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Relative error vs harmonics
    ax = axes[0, 1]
    rel_errors = [r['relative_error'] for r in results]
    ax.semilogy(n_harms, rel_errors, 'o-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Harmonics')
    ax.set_ylabel('Relative Error [%]')
    ax.set_title('Relative Reconstruction Error')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Original vs reconstructed (best case)
    ax = axes[1, 0]
    best_n_harm = n_harmonics_list[-1]  # Use highest
    fourier_best = compute_fourier_features(original, n_harmonics=best_n_harm)
    recon_best = reconstruct_from_fourier(fourier_best, npts, best_n_harm)
    
    theta_deg = np.linspace(0, 360, npts, endpoint=False)
    ax.plot(theta_deg, original[:, 0], 'b-', linewidth=2, label='Original Lateral', alpha=0.7)
    ax.plot(theta_deg, recon_best[:, 0], 'r--', linewidth=1.5, label=f'Recon Lateral (n={best_n_harm})')
    ax.set_xlabel('Angle [degrees]')
    ax.set_ylabel('Displacement [mm]')
    ax.set_title('Lateral Displacement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Original vs reconstructed tangential
    ax = axes[1, 1]
    ax.plot(theta_deg, original[:, 1], 'b-', linewidth=2, label='Original Tangential', alpha=0.7)
    ax.plot(theta_deg, recon_best[:, 1], 'r--', linewidth=1.5, label=f'Recon Tangential (n={best_n_harm})')
    ax.set_xlabel('Angle [degrees]')
    ax.set_ylabel('Displacement [mm]')
    ax.set_title('Tangential Displacement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fourier_reconstruction_quality.png', dpi=150)
    print("\n✓ Plot saved to 'fourier_reconstruction_quality.png'")
    
    return results


def test_with_real_wheel(env):

    
    print("\nReal Wheel Test:")
    print("="*60)
    
    state, info = env.reset()
    
    # Get Fourier representation
    tensionchanges = env.spoke_turns * env.adjustment_per_turn
    wheel_disp, tensions, fourier = env.wheel_calc(tensionchanges, return_fourier=True)
    
    print(f"Wheel displacement shape: {wheel_disp.shape}")
    print(f"Fourier features shape: {fourier.shape}")
    print(f"Number of harmonics: {env.n_harmonics}")
    
    # Check compression ratio
    original_size = len(wheel_disp)
    fourier_size = len(fourier)
    compression = original_size / fourier_size
    
    print(f"\nCompression: {original_size} -> {fourier_size} ({compression:.1f}x)")
    print(f"DC components: {fourier[:2]}")
    print(f"\nFirst 3 harmonic magnitudes (in-plane, lateral):")
    for k in range(1, min(4, env.n_harmonics + 1)):
        idx = 2 + 4*(k-1)
        print(f"  Harmonic {k}: in-plane={fourier[idx]:.4f}, lateral={fourier[idx+2]:.4f}")
    
    # Visualize the Fourier spectrum
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    harmonics = np.arange(1, env.n_harmonics + 1)
    
    # In-plane magnitudes
    in_plane_mags = [fourier[2 + 4*(k-1)] for k in harmonics]
    axes[0].bar(harmonics, in_plane_mags, alpha=0.7, color='blue')
    axes[0].set_xlabel('Harmonic Number')
    axes[0].set_ylabel('Magnitude [mm]')
    axes[0].set_title('In-plane (Tangential+Radial) Fourier Spectrum')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Lateral magnitudes
    lat_mags = [fourier[2 + 4*(k-1) + 2] for k in harmonics]
    axes[1].bar(harmonics, lat_mags, alpha=0.7, color='green')
    axes[1].set_xlabel('Harmonic Number')
    axes[1].set_ylabel('Magnitude [mm]')
    axes[1].set_title('Lateral Fourier Spectrum')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('fourier_spectrum.png', dpi=150)
    print("\n✓ Spectrum plot saved to 'fourier_spectrum.png'")
    
    return fourier


def test_fourier_features():

    
    # Create synthetic displacement with known harmonics
    npts = 360
    theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
    
    # Create a pattern: 2nd harmonic in tangential, 3rd in lateral
    tan = 5.0 * np.cos(2 * theta + 0.5)  # 2nd harmonic, phase 0.5
    lat = 3.0 * np.sin(3 * theta - 0.3)  # 3rd harmonic, phase -0.3
    
    # Stack into tot_def format (lateral, tangential)
    tot_def = np.column_stack([lat, tan])
    
    # Compute Fourier features
    features = compute_fourier_features(tot_def, n_harmonics=5)
    
    print("Feature vector shape:", features.shape)
    print("\nDC components:")
    print(f"  Mean tangential: {features[0]:.6f} (expected ~0)")
    print(f"  Mean lateral: {features[1]:.6f} (expected ~0)")
    
    # Check 2nd harmonic in-plane (should capture tangential component)
    k = 2
    idx_base = 2 + 4*(k-1)
    mag_in_plane = features[idx_base]
    phase_in_plane = features[idx_base + 1]
    
    print(f"\n2nd Harmonic in-plane:")
    print(f"  Magnitude: {mag_in_plane:.3f} (expected ~5.0)")
    print(f"  Phase: {phase_in_plane:.3f} (expected ~0.5)")
    
    # Check 3rd harmonic lateral
    k = 3
    idx_base = 2 + 4*(k-1)
    mag_lat = features[idx_base + 2]
    phase_lat = features[idx_base + 3]
    
    print(f"\n3rd Harmonic lateral:")
    print(f"  Magnitude: {mag_lat:.3f} (expected ~3.0)")
    print(f"  Phase: {phase_lat:.3f} (expected ~{-0.3 - np.pi/2:.3f}, shifted by -π/2 for sin)")
    
    return features


def test_rotation_invariance():

    
    npts = 360
    theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
    
    # Create original pattern
    tan = 5.0 * np.cos(2 * theta)
    lat = 2.0 * np.cos(3 * theta)
    
    tot_def_original = np.column_stack([lat, tan])
    features_original = compute_fourier_features(tot_def_original, n_harmonics=5)
    
    # Rotate by 45 degrees
    rotation = np.pi/4
    theta_rotated = theta + rotation
    tan_rot = 5.0 * np.cos(2 * theta_rotated)
    lat_rot = 2.0 * np.cos(3 * theta_rotated)
    
    tot_def_rotated = np.column_stack([lat_rot, tan_rot])
    features_rotated = compute_fourier_features(tot_def_rotated, n_harmonics=5)
    
    # Extract magnitudes (indices 2, 6, 10, 14, 18 for in-plane)
    # and (indices 4, 8, 12, 16, 20 for lateral)
    mags_original = []
    mags_rotated = []
    for k in range(1, 6):
        idx = 2 + 4*(k-1)
        mags_original.append(features_original[idx])      # in-plane
        mags_original.append(features_original[idx + 2])  # lateral
        mags_rotated.append(features_rotated[idx])
        mags_rotated.append(features_rotated[idx + 2])
    
    mags_original = np.array(mags_original)
    mags_rotated = np.array(mags_rotated)
    
    print("\nRotation Invariance Test:")
    print("Magnitude differences (should be near zero):")
    print(np.abs(mags_original - mags_rotated))
    print(f"Max difference: {np.max(np.abs(mags_original - mags_rotated)):.10f}")
    
    assert np.allclose(mags_original, mags_rotated, atol=1e-10), \
        "Magnitudes should be rotation-invariant!"
    print("✓ Magnitudes are rotation-invariant")


def test_computation_speed(n_resets=1000):

    import time
    
    print("\nSpeed Test: Comparing computation time")
    print("="*60)
    print(f"Running {n_resets} resets for each configuration...\n")
    
    # Test configurations
    configs = [
        ("rimpoints", 8),
        ("fourier", 8),
        ("fourier", 12),
        ("fourier", 20),
        ("fourier", 30),
        ("fourier", 40),
    ]
    
    results = []
    
    for state_space, n_harm in configs:
        env = WheelEnv(
            state_space_selection=state_space, 
            n_harmonics=n_harm,
            len_theta=360
        )
        
        # Warm-up
        for _ in range(10):
            env.reset()
        
        # Timed runs
        start = time.time()
        for _ in range(n_resets):
            state, info = env.reset()
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / n_resets) * 1000
        
        if state_space == "rimpoints":
            state_size = len(state)
            label = f"{state_space} (baseline)"
        else:
            state_size = len(state)
            label = f"{state_space} (n={n_harm})"
        
        results.append({
            'config': label,
            'state_size': state_size,
            'total_time': elapsed,
            'avg_time_ms': avg_time_ms,
            'state_space': state_space
        })
        
        print(f"{label:30s} | State size: {state_size:4d} | "
              f"Avg: {avg_time_ms:.3f} ms | Total: {elapsed:.2f}s")
    
    # Calculate overhead
    baseline_time = next(r['avg_time_ms'] for r in results if r['state_space'] == 'rimpoints')
    
    print("\n" + "="*60)
    print("Overhead Analysis:")
    print("="*60)
    
    for r in results:
        if r['state_space'] == 'fourier':
            overhead_ms = r['avg_time_ms'] - baseline_time
            overhead_pct = (overhead_ms / baseline_time) * 100
            compression = 200 / r['state_size']  # 200 is rimpoints state size
            
            print(f"\n{r['config']}:")
            print(f"  Overhead: {overhead_ms:.3f} ms ({overhead_pct:.1f}%)")
            print(f"  State compression: {compression:.1f}x")
            print(f"  Cost per feature reduction: {overhead_ms/(200-r['state_size']):.4f} ms")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Computation time
    ax = axes[0]
    configs_labels = [r['config'] for r in results]
    times = [r['avg_time_ms'] for r in results]
    colors = ['blue' if r['state_space'] == 'rimpoints' else 'orange' for r in results]
    
    bars = ax.bar(range(len(results)), times, color=colors, alpha=0.7)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(configs_labels, rotation=45, ha='right')
    ax.set_ylabel('Average Time per Reset [ms]')
    ax.set_title('Computation Time Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: State size vs computation time
    ax = axes[1]
    state_sizes = [r['state_size'] for r in results]
    
    # Separate baseline and fourier
    baseline_idx = [i for i, r in enumerate(results) if r['state_space'] == 'rimpoints']
    fourier_idx = [i for i, r in enumerate(results) if r['state_space'] == 'fourier']
    
    ax.scatter([state_sizes[i] for i in baseline_idx], 
              [times[i] for i in baseline_idx],
              s=100, c='blue', label='Rimpoints', alpha=0.7)
    ax.scatter([state_sizes[i] for i in fourier_idx], 
              [times[i] for i in fourier_idx],
              s=100, c='orange', label='Fourier', alpha=0.7)
    
    # Add labels for each point
    for i, r in enumerate(results):
        if r['state_space'] == 'fourier':
            n_harm = int(r['config'].split('n=')[1].strip(')'))
            ax.annotate(f'n={n_harm}', 
                       (state_sizes[i], times[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9)
    
    ax.set_xlabel('State Space Size (number of features)')
    ax.set_ylabel('Average Time per Reset [ms]')
    ax.set_title('State Size vs Computation Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fourier_speed_comparison.png', dpi=150)
    print("\n✓ Plot saved to 'fourier_speed_comparison.png'")
    
    return results


def test_training_speed_impact(n_steps=10000):

    import time
    
    print("\nTraining Speed Impact Estimation")
    print("="*60)
    print(f"Simulating {n_steps} environment steps...\n")
    
    configs = [
        ("rimpoints", 8),
        ("fourier", 8),
    ]
    
    for state_space, n_harm in configs:
        env = WheelEnv(
            state_space_selection=state_space,
            n_harmonics=n_harm,
            action_space_selection="continous"
        )
        
        # Simulate training loop
        start = time.time()
        
        state, info = env.reset()
        for _ in range(n_steps):
            # Random action
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                state, info = env.reset()
        
        elapsed = time.time() - start
        steps_per_sec = n_steps / elapsed
        
        label = "rimpoints" if state_space == "rimpoints" else f"fourier (n={n_harm})"
        print(f"{label:25s} | {steps_per_sec:8.1f} steps/s | "
              f"Total: {elapsed:.2f}s")
    
    print("\nEstimated time for 1M training steps:")
    for state_space, n_harm in configs:
        env = WheelEnv(
            state_space_selection=state_space,
            n_harmonics=n_harm,
            action_space_selection="continous"
        )
        
        # Quick timing
        start = time.time()
        state, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                state, info = env.reset()
        elapsed = time.time() - start
        
        time_per_1M = (elapsed / 1000) * 1_000_000
        label = "rimpoints" if state_space == "rimpoints" else f"fourier (n={n_harm})"
        print(f"  {label:25s}: ~{time_per_1M/60:.1f} minutes")


if __name__ == "__main__":
    print("="*60)
    print("Test 1: Synthetic data with known harmonics")
    print("="*60)
    test_fourier_features()
    
    print("\n" + "="*60)
    print("Test 2: Rotation invariance")
    print("="*60)
    test_rotation_invariance()
    
    print("\n" + "="*60)
    print("Test 3: Real wheel environment")
    print("="*60)
    env = WheelEnv(state_space_selection="fourier", n_harmonics=8)
    test_with_real_wheel(env)
    
    print("\n" + "="*60)
    print("Test 4: Reconstruction quality")
    print("="*60)
    # Test with various harmonics
    results = test_reconstruction_quality(env, n_harmonics_list=[2, 4, 6, 8, 10, 12, 16, 20])
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Reconstruction Quality")
    print("="*60)
    print(f"\nRecommendation based on results:")
    # Find the "elbow" - where we get 99% accuracy
    for r in results:
        if r['relative_error'] < 1.0:  # Less than 1% error
            print(f"✓ n_harmonics={r['n_harmonics']} achieves <1% error with {r['n_features']} features")
            break
    
    print("\n" + "="*60)
    print("Test 5: Computation Speed")
    print("="*60)
    speed_results = test_computation_speed(n_resets=1000)
    
    print("\n" + "="*60)
    print("Test 6: Training Speed Impact")
    print("="*60)
    test_training_speed_impact(n_steps=10000)
    
    print("\n" + "="*60)
    print("FINAL RECOMMENDATIONS")
    print("="*60)
    
    # Find best tradeoff
    print("\nBased on reconstruction quality and speed:")
    for r in results:
        if r['relative_error'] < 1.0:
            optimal_n = r['n_harmonics']
            optimal_features = r['n_features']
            break
    
    # Find corresponding speed result
    speed_result = next((s for s in speed_results 
                        if 'fourier' in s['config'] and f'n={optimal_n}' in s['config']), 
                       None)
    
    if speed_result:
        baseline = next(s for s in speed_results if s['state_space'] == 'rimpoints')
        overhead = speed_result['avg_time_ms'] - baseline['avg_time_ms']
        compression = baseline['state_size'] / optimal_features
        
        print(f"\n✓ Recommended: n_harmonics={optimal_n}")
        print(f"  • {optimal_features} features (vs {baseline['state_size']} for rimpoints)")
        print(f"  • {compression:.1f}x state compression")
        print(f"  • {overhead:.3f} ms overhead per reset ({overhead/baseline['avg_time_ms']*100:.1f}% slower)")
        print(f"  • <1% reconstruction error")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

"""