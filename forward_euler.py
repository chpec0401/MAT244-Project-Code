import numpy as np
def forward_euler(H, psi0, hbar, t0, dt, n_steps):
    M = np.eye(len(psi0)) - 1j * dt / hbar * H
    psi = np.zeros((n_steps + 1, len(psi0)), dtype=complex)
    psi[0] = psi0
    for n in range(n_steps):
        psi[n + 1] = M @ psi[n]
    times = t0 + dt * np.arange(n_steps + 1)
    return "forward_euler", times, psi
