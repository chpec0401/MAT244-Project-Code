def hermit(K, duration):
    ev, evec = np.linalg.eigh(K)
    phases = np.exp(-1j * duration * ev)

    return (
        evec * phases
    ) @ evec.conj().T

def solve_lie_trotter(problem):

    A = problem["A"]
    B_part = problem["g"] * problem["B"]

    psi0 = problem["psi0"]
    times = problem["times"]
    dt = problem["dt"]

    U_A = hermit(
        A,
        dt,
    )

    U_B = hermit(
        B_part,
        dt,
    )

    states = np.empty(
        (len(times), len(psi0)),
        dtype=np.complex128,
    )

    states[0] = psi0

    for n in range(len(times) - 1):
        states[n + 1] = U_A @ (
            U_B @ states[n]
        )

    return {
        "method": "lie_trotter",
        "times": times.copy(),
        "states": states,

    }