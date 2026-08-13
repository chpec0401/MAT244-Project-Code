import numpy as np
from scipy.linalg import expm
def strang(H_A, H_B, psi0, hbar, t0, dt, n_steps):
    U_half = expm(-1j * (dt / 2) * H_A / hbar)
    U_B = expm(-1j * dt * H_B / hbar)
    psi = np.zeros((n_steps + 1, len(psi0)), dtype=complex)
    psi[0] = psi0
    for n in range(n_steps):
        psi[n + 1] = U_half @ (U_B @ (U_half @ psi[n]))
    times = t0 + dt * np.arange(n_steps + 1)
    return "strang", times, psi
