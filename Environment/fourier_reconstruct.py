"""
Fourier Mode Reconstruction Analysis - For Use with Actual WheelEnv

Run this script in your environment where bikewheelcalc is installed.
It will use your actual wheel simulation to generate displacement data.
"""

import numpy as np
import matplotlib.pyplot as plt
from wheel_env import WheelEnv  # Adjust import path as needed


def compute_fourier_coefficients(signal: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Compute Fourier coefficients for a signal."""
    npts = len(signal)
    fft = np.fft.fft(signal) / npts
    
    coeffs = [np.real(fft[0])]
    for k in range(1, n_harmonics + 1):
        coeffs.append(np.abs(fft[k]) * 2)
        coeffs.append(np.angle(fft[k]))
    
    return np.array(coeffs)


def reconstruct_from_fourier(coeffs: np.ndarray, npts: int) -> np.ndarray:
    """Reconstruct signal from Fourier coefficients."""
    signal = np.zeros(npts)
    theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
    
    signal += coeffs[0]
    
    n_harmonics = (len(coeffs) - 1) // 2
    for k in range(1, n_harmonics + 1):
        mag = coeffs[2*k - 1]
        phase = coeffs[2*k]
        signal += mag * np.cos(k * theta + phase)
    
    return signal


def analyze_single_state(radial, lateral, max_harmonics=50):
    """Compute MSE and R² for a single wheel state."""
    npts = len(radial)
    
    mse_rad = np.zeros(max_harmonics)
    mse_lat = np.zeros(max_harmonics)
    
    for n_harm in range(1, max_harmonics + 1):
        # Radial
        coeffs_rad = compute_fourier_coefficients(radial, n_harm)
        recon_rad = reconstruct_from_fourier(coeffs_rad, npts)
        mse_rad[n_harm - 1] = np.mean((radial - recon_rad)**2)
        
        # Lateral
        coeffs_lat = compute_fourier_coefficients(lateral, n_harm)
        recon_lat = reconstruct_from_fourier(coeffs_lat, npts)
        mse_lat[n_harm - 1] = np.mean((lateral - recon_lat)**2)
    
    return mse_rad, mse_lat


def run_full_analysis(n_samples=50, max_harmonics=50, random_spokes_range=[3, 5, 8, 12]):
    """
    Run comprehensive analysis across multiple random wheel states.
    Tests different levels of wheel damage (number of affected spokes).
    """
    results = {}
    
    for n_random in random_spokes_range:
        print(f"\nAnalyzing with {n_random} random spokes affected...")
        
        env = WheelEnv(
            state_space_selection="rimpoints",
            len_theta=360,
            n_spokes=36,
            random_spoke_n=n_random,
            random_spoke_turns_max=1.0,
            include_tan_displacement=False
        )
        
        all_mse_rad = []
        all_mse_lat = []
        all_var_rad = []
        all_var_lat = []
        
        for i in range(n_samples):
            obs, info = env.reset()
            
            displacement = obs.reshape(-1, 2)
            radial = displacement[:, 0]
            lateral = displacement[:, 1]
            
            mse_rad, mse_lat = analyze_single_state(radial, lateral, max_harmonics)
            
            all_mse_rad.append(mse_rad)
            all_mse_lat.append(mse_lat)
            all_var_rad.append(np.var(radial))
            all_var_lat.append(np.var(lateral))
        
        results[n_random] = {
            'mean_mse_rad': np.mean(all_mse_rad, axis=0),
            'mean_mse_lat': np.mean(all_mse_lat, axis=0),
            'std_mse_rad': np.std(all_mse_rad, axis=0),
            'std_mse_lat': np.std(all_mse_lat, axis=0),
            'mean_var_rad': np.mean(all_var_rad),
            'mean_var_lat': np.mean(all_var_lat),
        }
        
        # Compute R²
        results[n_random]['r2_rad'] = 1 - results[n_random]['mean_mse_rad'] / results[n_random]['mean_var_rad']
        results[n_random]['r2_lat'] = 1 - results[n_random]['mean_mse_lat'] / results[n_random]['mean_var_lat']
        results[n_random]['r2_combined'] = (results[n_random]['r2_rad'] + results[n_random]['r2_lat']) / 2
        
        env.close()
    
    return results


def plot_results(results, save_path='fourier_analysis_wheelenv.png'):
    """Create visualization with MSE plots."""
    harmonics = np.arange(1, 51)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
    
    # Plot 1: Combined MSE vs harmonics (log scale)
    ax1 = axes[0]
    for (n_random, data), color in zip(results.items(), colors):
        combined_mse = data['mean_mse_rad'] + data['mean_mse_lat']
        ax1.semilogy(harmonics, combined_mse, color=color, linewidth=2, 
                     label=f'{n_random} spokes')
    ax1.axvline(x=30, color='red', linestyle='--', alpha=0.7, linewidth=2, label='n=30')
    ax1.set_xlabel('Number of Fourier Harmonics', fontsize=12)
    ax1.set_ylabel('Combined MSE [mm²]', fontsize=12)
    ax1.set_title('Reconstruction Error vs. Number of Harmonics', fontsize=14)
    ax1.legend(title='Random spokes')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Radial vs Lateral Reconstruction Percentage
    ax2 = axes[1]
    mid_key = list(results.keys())[len(results)//2]
    data = results[mid_key]
    
    # Calculate reconstruction percentage: (1 - MSE/Variance) * 100
    recon_pct_rad = (1 - data['mean_mse_rad'] / data['mean_var_rad']) * 100
    recon_pct_lat = (1 - data['mean_mse_lat'] / data['mean_var_lat']) * 100
    
    # Propagate uncertainty
    std_pct_rad = (data['std_mse_rad'] / data['mean_var_rad']) * 100
    std_pct_lat = (data['std_mse_lat'] / data['mean_var_lat']) * 100
    
    ax2.plot(harmonics, recon_pct_rad, 'b-', linewidth=2, label='Radial')
    ax2.plot(harmonics, recon_pct_lat, 'r-', linewidth=2, label='Lateral')
    ax2.fill_between(harmonics, 
                     recon_pct_rad - std_pct_rad,
                     recon_pct_rad + std_pct_rad,
                     color='blue', alpha=0.2)
    ax2.fill_between(harmonics,
                     recon_pct_lat - std_pct_lat,
                     recon_pct_lat + std_pct_lat,
                     color='red', alpha=0.2)
    ax2.axvline(x=30, color='green', linestyle='--', alpha=0.7, linewidth=2, label='n=30')
    ax2.axhline(y=99, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=99.9, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Number of Fourier Harmonics', fontsize=12)
    ax2.set_ylabel('Reconstruction Accuracy [%]', fontsize=12)
    ax2.set_title(f'Radial vs Lateral Reconstruction ({mid_key} random spokes)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([80, 100.5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    
    return fig


def print_latex_table(results):
    """Print results as LaTeX table for thesis."""
    print("\n" + "="*80)
    print("LATEX TABLE FOR THESIS")
    print("="*80)
    
    print(r"""
\begin{table}[htbp]
\centering
\caption{Fourier reconstruction accuracy for different numbers of harmonics}
\label{tab:fourier_modes}
\begin{tabular}{ccccc}
\toprule
Harmonics & Features & MSE$_{\text{rad}}$ [mm²] & MSE$_{\text{lat}}$ [mm²] & $R^2$ [\%] \\
\midrule""")
    
    # Use middle case
    mid_key = list(results.keys())[len(results)//2]
    data = results[mid_key]
    
    key_harmonics = [1, 2, 4, 8, 10, 15, 20, 25, 30, 40, 50]
    for h in key_harmonics:
        idx = h - 1
        n_features = 2 + 4*h
        r2 = data['r2_combined'][idx] * 100
        print(f"{h} & {n_features} & {data['mean_mse_rad'][idx]:.4f} & "
              f"{data['mean_mse_lat'][idx]:.4f} & {r2:.2f} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def main():
    print("="*80)
    print("FOURIER MODE RECONSTRUCTION ANALYSIS FOR WHEEL TRUING ENVIRONMENT")
    print("="*80)
    
    # Run analysis with different damage levels
    results = run_full_analysis(
        n_samples=50,
        max_harmonics=50,
        random_spokes_range=[3, 5, 8, 12]
    )
    
    # Print summary for each condition
    for n_random, data in results.items():
        print(f"\n--- Results for {n_random} random spokes ---")
        print(f"{'Harmonics':<12} {'MSE Combined':>14} {'R² Combined':>12}")
        print("-" * 40)
        for h in [5, 10, 15, 20, 25, 30, 40]:
            idx = h - 1
            mse = data['mean_mse_rad'][idx] + data['mean_mse_lat'][idx]
            r2 = data['r2_combined'][idx] * 100
            print(f"{h:<12} {mse:>14.6f} {r2:>11.4f}%")
    
    # Create visualization
    plot_results(results)
    
    # Print LaTeX table
    print_latex_table(results)
    
    # Final recommendation
    print("\n" + "="*80)
    print("ANALYSIS CONCLUSION")
    print("="*80)
    
    # Find optimal number of harmonics
    mid_key = list(results.keys())[len(results)//2]
    r2_combined = results[mid_key]['r2_combined']
    
    for thresh in [99.0, 99.5, 99.9]:
        idx = np.argmax(r2_combined * 100 >= thresh)
        if r2_combined[idx] * 100 >= thresh:
            print(f"Minimum harmonics for R² ≥ {thresh}%: {idx + 1} ({2 + 4*(idx+1)} features)")
    
    print(f"\nYour choice of n_harmonics=30 achieves R² = {r2_combined[29]*100:.2f}%")
    print(f"This represents a {720/(2+4*30):.1f}x compression of the state space.")


if __name__ == "__main__":
    main()