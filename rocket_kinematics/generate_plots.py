import numpy as np
import matplotlib.pyplot as plt

# Presentation styling
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 22
})

# Physics params
m0 = 1000
m_prime = 20
vex = 100
t_burn = 40
t = np.linspace(0, t_burn, 400)
t1 = 30

def v(t):
    return -vex * np.log(1 - (m_prime/m0)*t)

def x(t):
    term = 1 - (m_prime/m0)*t
    return vex * (t + (m0/m_prime) * term * np.log(term))

# --- Plot 1: Position x(t) ---
plt.figure(figsize=(10, 6))
plt.plot(t, x(t), color='#2980b9', linewidth=4)
plt.plot(t1, x(t1), 'ro', markersize=10, label=f'$t_1$ = {t1}s')
plt.vlines(t1, 0, x(t1), colors='red', linestyles='dashed', linewidth=2.5)
plt.hlines(x(t1), 0, t1, colors='red', linestyles='dashed', linewidth=2.5, alpha=0.5)

plt.title('Position vs Time')
plt.xlabel('Time $t$ (s)')
plt.ylabel('Position $x(t)$ (m)')
plt.xlim(0, t_burn)
plt.ylim(0, x(t_burn)*1.05)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_1_position.png', dpi=300)
plt.close()

# --- Plot 2: Velocity v(t) Analytic ---
plt.figure(figsize=(10, 6))
plt.plot(t, v(t), color='#f39c12', linewidth=4)
t_fill = np.linspace(0, t1, 100)
plt.fill_between(t_fill, v(t_fill), color='#f39c12', alpha=0.3)

plt.title('Analytic Integration (Area under Curve)')
plt.xlabel('Time $t$ (s)')
plt.ylabel('Velocity $v(t)$ (m/s)')
plt.xlim(0, t_burn)
plt.ylim(0, v(t_burn)*1.05)

# Text annotation for area
plt.text(t1/2, v(t_burn)*0.75, r'$x(t_1) = x_0 + \int_0^{t_1} v(t) dt$', horizontalalignment='center', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_2_v_analytic.png', dpi=300)
plt.close()

# --- Plot 3: Velocity v(t) Numeric (Rectangles) ---
plt.figure(figsize=(10, 6))
plt.plot(t, v(t), color='#f39c12', linewidth=4)

# Riemann sum rectangles (Euler method visualization)
dt = 3  # width of each rectangle in seconds
t_rects = np.arange(0, t1, dt)
v_rects = v(t_rects)

plt.bar(t_rects, v_rects, width=dt, align='edge', color='#2ecc71', edgecolor='black', alpha=0.7, linewidth=2)

plt.title('Numerical Integration (Euler Method)')
plt.xlabel('Time $t$ (s)')
plt.ylabel('Velocity $v(t)$ (m/s)')
plt.xlim(0, t_burn)
plt.ylim(0, v(t_burn)*1.05)

# Text annotation for sum
plt.text(t1/2, v(t_burn)*0.75, r'$x(t_1) \approx x_0 + \sum v_i \Delta t$', horizontalalignment='center', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_3_v_numeric.png', dpi=300)
plt.close()

print("Plots generated successfully!")
