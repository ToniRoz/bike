import numpy as np
import matplotlib.pyplot as plt
import time

from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix


# ============================================================
# PARAMETERS
# ============================================================
LEN_THETA = 360
N_SPOKES = 36
MAX_MODES = 100
MIN_MODES = 3

N_TIME_REPEATS = 20

INIT_TENSION = 800.0
ADJUSTMENT_PER_TURN = 25.4 / 56 / 1000  # m

SEED = 123
np.random.seed(SEED)


# ============================================================
# FIXED SPOKE CONFIGURATION (ONCE)
# ============================================================
spoke_turns = np.zeros(N_SPOKES)
idx = np.random.choice(N_SPOKES, 5, replace=False)
spoke_turns[idx] = np.random.rand(5) - 0.5

tension_changes = spoke_turns * ADJUSTMENT_PER_TURN


# ============================================================
# WHEEL FACTORY (DETERMINISTIC)
# ============================================================
def make_wheel():
    wheel = BicycleWheel()
    wheel.hub = Hub(width=0.05, diameter=0.04)
    wheel.rim = Rim(
        radius=0.3,
        area=100e-6,
        I_lat=1500e-12,
        I_rad=3000e-12,
        J_tor=500e-12,
        I_warp=0.0,
        young_mod=69e9,
        shear_mod=26e9,
    )
    wheel.lace_cross(
        n_spokes=N_SPOKES,
        n_cross=3,
        diameter=2.0e-3,
        young_mod=210e9,
    )
    wheel.apply_tension(INIT_TENSION)
    return wheel


# ============================================================
# SOLVER
# ============================================================
def solve_rim(N_modes):
    wheel = make_wheel()
    mm = ModeMatrix(wheel, N=N_modes)

    theta = np.linspace(-np.pi, np.pi, LEN_THETA)

    B_lat = mm.B_theta(theta, 0)
    B_rad = mm.B_theta(theta, 1)
    B_tan = mm.B_theta(theta, 2)

    K = mm.K_rim(tension=True) + mm.K_spk(
        smeared_spokes=False, tension=True
    )
    A = mm.A_adj()

    F = A @ tension_changes
    dm = np.linalg.solve(K, F)

    rim = np.stack(
        [
            B_rad @ dm,
            B_lat @ dm,
            B_tan @ dm,
        ],
        axis=1,
    )

    return rim * 1000.0  # mm


# ============================================================
# REFERENCE SOLUTION (ONCE)
# ============================================================
print("Computing reference (100 modes)")
rim_ref = solve_rim(MAX_MODES)


# ============================================================
# MODE SWEEP
# ============================================================
modes = []
mse_rad = []
mse_lat = []
mse_tan = []
mean_time = []

for N in range(MAX_MODES - 1, MIN_MODES - 1, -1):
    print(f"Modes: {N}")

    # ---- error (deterministic, once)
    rim_N = solve_rim(N)
    diff = rim_N - rim_ref

    mse_rad.append(np.mean(diff[:, 0] ** 2))
    mse_lat.append(np.mean(diff[:, 1] ** 2))
    mse_tan.append(np.mean(diff[:, 2] ** 2))
    modes.append(N)

    # ---- timing (repeat)
    times = []
    for _ in range(N_TIME_REPEATS):
        t0 = time.perf_counter()
        solve_rim(N)
        times.append(time.perf_counter() - t0)

    mean_time.append(np.mean(times))


# ============================================================
# PLOT
# ============================================================
fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.set_yscale("log")
ax1.plot(modes, mse_rad, label="Radial MSE")
ax1.plot(modes, mse_lat, label="Lateral MSE")
ax1.plot(modes, mse_tan, label="Tangential MSE")

ax1.set_xlabel("Number of modes")
ax1.set_ylabel("Mean squared error vs 100-mode solution [mm²]")
ax1.grid(True, which="both")
ax1.invert_xaxis()

ax2 = ax1.twinx()
ax2.plot(modes, mean_time, linestyle="--", color="k", label="Solve time")
ax2.set_ylabel("Mean solve time [s]")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Modal truncation error and solve time (fixed spoke configuration)")
plt.tight_layout()
plt.savefig("mode_truncation_error_time_fixed_spokes.png", dpi=200)
plt.show()

print("Saved plot: mode_truncation_error_time_fixed_spokes.png")
