"""
MAT244 - exact matrix exponential and RK4.

Solves i*hbar*dpsi/dt = H*psi for H = H_A + H_B with hbar = 1, using
the exact matrix exponential and classical RK4, and compares them.

Main test:   H_A = Z, H_B = 0.7*X, psi0 = (1,0)^T
Long run:    T = 10, dt = 0.1
Convergence: T = 2, n_steps = 10, 20, 40, 80, 160, 320
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm



# pauli matrices
I2 = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0],
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0],
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1],
], dtype=complex)

np.set_printoptions(precision=8, suppress=True)

SAVE_FIGURES = True


# problem setup
def make_problem(problem_id, H_A, H_B, psi0, hbar, t0, t_final, n_steps):
    H_A = np.array(H_A, dtype=complex)
    H_B = np.array(H_B, dtype=complex)
    psi0 = np.array(psi0, dtype=complex)

    H = H_A + H_B
    dt = (t_final - t0) / n_steps
    times = np.linspace(t0, t_final, n_steps + 1)

    problem = {}
    problem["problem_id"] = problem_id
    problem["H_A"] = H_A
    problem["H_B"] = H_B
    problem["H"] = H
    problem["psi0"] = psi0
    problem["hbar"] = hbar
    problem["t0"] = t0
    problem["t_final"] = t_final
    problem["n_steps"] = n_steps
    problem["dt"] = dt
    problem["times"] = times
    problem["dimension"] = psi0.size
    return problem


def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)


def validate_problem(problem):
    H_A = problem["H_A"]
    H_B = problem["H_B"]
    H = problem["H"]
    psi0 = problem["psi0"]

    problems_found = []

    if H_A.shape != H_B.shape:
        problems_found.append("H_A and H_B have different shapes")

    if H.shape[0] != H.shape[1]:
        problems_found.append("H is not square")

    if psi0.shape != (problem["dimension"],):
        problems_found.append("psi0 has the wrong shape")

    if not is_hermitian(H_A):
        problems_found.append("H_A is not Hermitian")

    if not is_hermitian(H_B):
        problems_found.append("H_B is not Hermitian")

    if not is_hermitian(H):
        problems_found.append("H is not Hermitian")

    if not np.allclose(H, H_A + H_B):
        problems_found.append("H is not equal to H_A + H_B")

    if problem["hbar"] <= 0:
        problems_found.append("hbar must be positive")

    if not np.isclose(np.vdot(psi0, psi0).real, 1.0):
        problems_found.append("psi0 is not normalised")

    if len(problem["times"]) != problem["n_steps"] + 1:
        problems_found.append("wrong number of stored times")

    if len(problems_found) == 0:
        print("problem setup OK:", problem["problem_id"])
        return True
    else:
        print("problem setup FAILED:", problem["problem_id"])
        for message in problems_found:
            print("   -", message)
        return False


def validate_method_output(problem, result, expected_method_id):
    states = result["states"]
    problems_found = []

    if result["method_id"] != expected_method_id:
        problems_found.append("method_id is not " + expected_method_id)

    if not np.array_equal(result["times"], problem["times"]):
        problems_found.append("the time grid was changed")

    if states.shape != (problem["n_steps"] + 1, problem["dimension"]):
        problems_found.append("states has the wrong shape")

    if not np.iscomplexobj(states):
        problems_found.append("states is not complex")

    if not np.allclose(states[0], problem["psi0"]):
        problems_found.append("states[0] is not psi0")

    if not np.all(np.isfinite(states)):
        problems_found.append("states contains inf or nan")

    if len(problems_found) == 0:
        print(expected_method_id, "output contract passed")
        return True
    else:
        print(expected_method_id, "output contract FAILED")
        for message in problems_found:
            print("   -", message)
        return False



# the two methods
def exact_trajectory(problem):
    # psi(t_n) = exp(-i (t_n - t0) H / hbar) psi0.
    # H is Hermitian so -iH/hbar is skew-Hermitian, which makes the
    # propagator unitary. This is the reference for every state error.
    H = problem["H"]
    psi0 = problem["psi0"]
    hbar = problem["hbar"]
    t0 = problem["t0"]
    times = problem["times"]
    n_steps = problem["n_steps"]
    dimension = problem["dimension"]

    states = np.zeros((n_steps + 1, dimension), dtype=complex)

    for n in range(n_steps + 1):
        tau = times[n] - t0
        U = expm(-1j * tau * H / hbar)
        states[n] = U @ psi0

    return {
        "method_id": "exact",
        "times": times,
        "states": states,
    }


def pauli_closed_form_trajectory(problem, g):
    # for H = Z + gX we have H^2 = (1 + g^2) I, so with
    # omega = sqrt(1 + g^2)
    #   exp(-i tau H) = cos(omega tau) I - i sin(omega tau) H / omega
    H = problem["H"]
    psi0 = problem["psi0"]
    hbar = problem["hbar"]
    t0 = problem["t0"]
    times = problem["times"]
    n_steps = problem["n_steps"]
    dimension = problem["dimension"]

    omega = np.sqrt(1.0 + g**2)
    states = np.zeros((n_steps + 1, dimension), dtype=complex)

    for n in range(n_steps + 1):
        tau = times[n] - t0
        U = (
            np.cos(omega * tau / hbar) * I2
            - 1j * np.sin(omega * tau / hbar) * H / omega
        )
        states[n] = U @ psi0

    return states


def rk4_trajectory(problem):
    # The k_j are derivatives, so they do not already include dt.
    H = problem["H"]
    psi0 = problem["psi0"]
    hbar = problem["hbar"]
    dt = problem["dt"]
    n_steps = problem["n_steps"]
    dimension = problem["dimension"]

    states = np.zeros((n_steps + 1, dimension), dtype=complex)
    states[0] = psi0

    def f(psi):
        return -(1j / hbar) * (H @ psi)

    for n in range(n_steps):
        psi_n = states[n]

        k1 = f(psi_n)
        k2 = f(psi_n + 0.5 * dt * k1)
        k3 = f(psi_n + 0.5 * dt * k2)
        k4 = f(psi_n + dt * k3)

        states[n + 1] = psi_n + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return {
        "method_id": "rk4",
        "times": problem["times"],
        "states": states,
    }


# diagnostics

def squared_norms(states):
    n_times = len(states)
    dimension = states.shape[1]
    values = np.zeros(n_times)

    for n in range(n_times):
        total = 0.0 + 0.0j
        for j in range(dimension):
            total = total + np.conj(states[n][j]) * states[n][j]
        values[n] = total.real

    return values


def norm_errors(states):
    return np.abs(squared_norms(states) - 1.0)


def state_errors(states_num, states_exact):
    return np.linalg.norm(states_num - states_exact, axis=1)


def final_state_error(states_num, states_exact):
    return np.linalg.norm(states_num[-1] - states_exact[-1])


def normalized_energy_expectation(states, H):
    # E_n = (psi^dag H psi) / (psi^dag psi)
    n_times = len(states)
    dimension = states.shape[1]

    H_applied = states @ H.T

    values = np.zeros(n_times)
    for n in range(n_times):
        top = 0.0 + 0.0j
        bottom = 0.0 + 0.0j
        for j in range(dimension):
            top = top + np.conj(states[n][j]) * H_applied[n][j]
            bottom = bottom + np.conj(states[n][j]) * states[n][j]
        values[n] = (top / bottom).real

    return values


def rk4_one_step_matrix(problem):
    # R4(M) = I + M + M^2/2! + M^3/3! + M^4/4!
    # cut off after the fourth term.
    H = problem["H"]
    dt = problem["dt"]
    hbar = problem["hbar"]
    dimension = problem["dimension"]

    I = np.eye(dimension, dtype=complex)
    M = -1j * dt * H / hbar

    M2 = M @ M
    M3 = M2 @ M
    M4 = M3 @ M

    return I + M + M2 / 2.0 + M3 / 6.0 + M4 / 24.0


def exact_one_step_matrix(problem):
    return expm(-1j * problem["dt"] * problem["H"] / problem["hbar"])


def unitarity_defect(step_matrix):
    # Zero exactly when the step matrix is unitary.
    dimension = step_matrix.shape[0]
    I = np.eye(dimension, dtype=complex)
    product = step_matrix.conj().T @ step_matrix
    return np.linalg.norm(product - I, ord="fro")


def observed_orders(errors):
    orders = [np.nan]
    for j in range(len(errors) - 1):
        ratio = errors[j] / errors[j + 1]
        orders.append(np.log(ratio) / np.log(2.0))
    return orders


def reduction_factors(errors):
    factors = []
    for j in range(len(errors) - 1):
        factors.append(errors[j] / errors[j + 1])
    return np.array(factors)


def save_and_show(filename):
    if SAVE_FIGURES:
        plt.savefig(filename, dpi=200)
    plt.show()



# main problem: H = Z + 0.7X, T = 10, dt = 0.1


G_MAIN = 0.7

main_problem = make_problem(
    problem_id="main_noncommuting_long",
    H_A=Z,
    H_B=G_MAIN * X,
    psi0=np.array([1, 0], dtype=complex),
    hbar=1.0,
    t0=0.0,
    t_final=10.0,
    n_steps=100,
)

validate_problem(main_problem)

print("\nH =\n", main_problem["H"])
print("psi0 =", main_problem["psi0"])
print("dt =", main_problem["dt"])
print("n_steps =", main_problem["n_steps"])
print("stored states =", main_problem["n_steps"] + 1)

exact_result = exact_trajectory(main_problem)
rk4_result = rk4_trajectory(main_problem)

validate_method_output(main_problem, exact_result, "exact")
validate_method_output(main_problem, rk4_result, "rk4")

closed_form_states = pauli_closed_form_trajectory(main_problem, G_MAIN)
closed_form_difference = np.max(
    np.linalg.norm(exact_result["states"] - closed_form_states, axis=1)
)

print("\nMax difference, expm vs Pauli closed form:", closed_form_difference)



#errors over the main run

exact_states = exact_result["states"]
rk4_states = rk4_result["states"]
times = main_problem["times"]

exact_norm_error = norm_errors(exact_states)
rk4_norm_error = norm_errors(rk4_states)
rk4_state_error = state_errors(rk4_states, exact_states)

exact_energy = normalized_energy_expectation(exact_states, main_problem["H"])
rk4_energy = normalized_energy_expectation(rk4_states, main_problem["H"])

E0 = exact_energy[0]
exact_energy_error = np.abs(exact_energy - E0)
rk4_energy_error = np.abs(rk4_energy - E0)

print("\nRK4 final-time state error =", rk4_state_error[-1])
print("RK4 final norm error       =", rk4_norm_error[-1])
print("Exact max norm error       =", exact_norm_error.max())
print("RK4 max energy error       =", rk4_energy_error.max())
print("Exact max energy error     =", exact_energy_error.max())
print("RK4 norm err / exact max   =",
      rk4_norm_error[-1] / exact_norm_error.max())



# Plots
floor = np.finfo(float).eps

exact_probabilities = np.abs(exact_states)**2
rk4_probabilities = np.abs(rk4_states)**2

plt.figure(figsize=(9, 5.5))
plt.plot(times, exact_probabilities[:, 0], label=r"Exact $|\psi_0(t)|^2$")
plt.plot(times, exact_probabilities[:, 1], label=r"Exact $|\psi_1(t)|^2$")
plt.scatter(times, rk4_probabilities[:, 0], s=14, label=r"RK4 $|\psi_0(t)|^2$")
plt.scatter(times, rk4_probabilities[:, 1], s=14, label=r"RK4 $|\psi_1(t)|^2$")
plt.xlabel("Time")
plt.ylabel("Basis-state probability")
plt.title(r"Exact and RK4 evolution: $H=Z+0.7X$, $dt=0.1$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
save_and_show("rk4_probabilities.png")

plt.figure(figsize=(9, 5.5))
plt.semilogy(times, np.maximum(exact_norm_error, floor),
             label="Exact matrix exponential")
plt.semilogy(times, np.maximum(rk4_norm_error, floor), label="RK4")
plt.xlabel("Time")
plt.ylabel(r"$|\|\psi_n\|_2^2-1|$")
plt.title("Norm error versus time")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
save_and_show("rk4_norm_time.png")

plt.figure(figsize=(9, 5.5))
plt.semilogy(times, np.maximum(rk4_state_error, floor),
             label="RK4 state error")
plt.xlabel("Time")
plt.ylabel(r"$\|\psi_{\mathrm{RK4}}(t_n)-\psi_{\mathrm{exact}}(t_n)\|_2$")
plt.title("RK4 raw state error versus time")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
save_and_show("rk4_state_error_time.png")

plt.figure(figsize=(9, 5.5))
plt.semilogy(times, np.maximum(exact_energy_error, floor), label="Exact")
plt.semilogy(times, np.maximum(rk4_energy_error, floor), label="RK4")
plt.xlabel("Time")
plt.ylabel(r"$|E_n-E_0|$")
plt.title("Normalized energy-expectation error versus time")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
save_and_show("rk4_energy_time.png")


# one step matrix comparison at dt = 0.1

R_rk4 = rk4_one_step_matrix(main_problem)
U_exact = exact_one_step_matrix(main_problem)

print("\nRK4 one-step matrix =\n", R_rk4)
print("Exact one-step matrix =\n", U_exact)
print("||R_RK4 - U_exact||_F =", np.linalg.norm(R_rk4 - U_exact, ord="fro"))
print("RK4 unitarity defect =", unitarity_defect(R_rk4))
print("Exact unitarity defect =", unitarity_defect(U_exact))


# Convergence at fixed T = 2

CONVERGENCE_N_STEPS = [10, 20, 40, 80, 160, 320]

rows = []
for N in CONVERGENCE_N_STEPS:
    problem = make_problem(
        problem_id="rk4_convergence_N" + str(N),
        H_A=Z,
        H_B=G_MAIN * X,
        psi0=np.array([1, 0], dtype=complex),
        hbar=1.0,
        t0=0.0,
        t_final=2.0,
        n_steps=N,
    )

    exact = exact_trajectory(problem)
    rk4 = rk4_trajectory(problem)

    rows.append({
        "n_steps": N,
        "dt": problem["dt"],
        "final_state_error": final_state_error(rk4["states"], exact["states"]),
        "final_norm_error": norm_errors(rk4["states"])[-1],
    })

convergence_table = pd.DataFrame(rows)

state_errors_array = convergence_table["final_state_error"].to_numpy()
norm_errors_array = convergence_table["final_norm_error"].to_numpy()

convergence_table["observed_order"] = observed_orders(state_errors_array)
convergence_table["norm_observed_order"] = observed_orders(norm_errors_array)

print("\nConvergence table")
print(convergence_table.to_string(index=False))

dt_values = convergence_table["dt"].to_numpy()
errors = convergence_table["final_state_error"].to_numpy()

# O(dt^4) reference line anchored to the finest-grid RK4 error.
reference = errors[-1] * (dt_values / dt_values[-1])**4

plt.figure(figsize=(9, 5.5))
plt.loglog(dt_values, errors, marker="o",
           label="Measured RK4 final-time error")
plt.loglog(dt_values, reference, linestyle="--", label=r"Reference $O(dt^4)$")
plt.xlabel(r"Step size $dt$")
plt.ylabel(r"$\|\psi_N^{RK4}-\psi^{exact}(T)\|_2$")
plt.title(r"RK4 convergence at fixed final time $T=2$")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
save_and_show("rk4_convergence.png")

plt.figure(figsize=(9, 5.5))
plt.loglog(convergence_table["dt"],
           np.maximum(convergence_table["final_norm_error"], floor),
           marker="o")
plt.xlabel(r"Step size $dt$")
plt.ylabel(r"$|\|\psi_N\|_2^2-1|$")
plt.title("RK4 final-time norm error versus step size")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
save_and_show("rk4_norm_convergence.png")


valid_orders = convergence_table["observed_order"].dropna()
norm_orders = convergence_table["norm_observed_order"].dropna()

print("\nObserved orders, state error:", valid_orders.to_numpy())
print("Mean:", valid_orders.mean())
print("State-error reduction factors:", reduction_factors(state_errors_array))

print("\nObserved orders, norm error:", norm_orders.to_numpy())
print("Mean:", norm_orders.mean())
print("Norm-error reduction factors:", reduction_factors(norm_errors_array))


zero_problem = make_problem(
    problem_id="zero_H",
    H_A=np.zeros((2, 2), dtype=complex),
    H_B=np.zeros((2, 2), dtype=complex),
    psi0=np.array([1, 0], dtype=complex),
    hbar=1.0,
    t0=0.0,
    t_final=2.0,
    n_steps=20,
)

zero_exact = exact_trajectory(zero_problem)
zero_rk4 = rk4_trajectory(zero_problem)

print("\nExact zero-H max deviation:",
      np.max(np.linalg.norm(zero_exact["states"] - zero_problem["psi0"],
                            axis=1)))
print("RK4 zero-H max deviation:",
      np.max(np.linalg.norm(zero_rk4["states"] - zero_problem["psi0"],
                            axis=1)))

complex_problem = make_problem(
    problem_id="complex_hermitian",
    H_A=0.2 * I2 + 0.8 * Z,
    H_B=0.5 * X - 0.3 * Y,
    psi0=np.array([1, 1j], dtype=complex) / np.sqrt(2),
    hbar=1.0,
    t0=0.0,
    t_final=2.0,
    n_steps=40,
)

validate_problem(complex_problem)

complex_exact = exact_trajectory(complex_problem)
complex_rk4 = rk4_trajectory(complex_problem)

print("Complex test RK4 final state error:",
      final_state_error(complex_rk4["states"], complex_exact["states"]))
print("Complex test RK4 final norm error:",
      norm_errors(complex_rk4["states"])[-1])