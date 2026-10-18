"""Validate quasi-one-dimensional nozzle computations against Aerokit.

The case uses the smooth converging-diverging area law from the nozzle
examples.  ``flowdyn.solution.euler_nozzle.nozzle`` delegates the reference
isentropic/shock construction to Aerokit, while Flowdyn evolves the finite-
volume discretization to a steady state.

Two pressure ratios are deliberately retained:

* ``NPR=1.05`` is wholly subsonic, so the reference contains no discontinuity;
* ``NPR=1.10`` is choked with a normal shock in the divergent part.  Its
  discontinuity limits the global order, which is expected rather than a
  regression.

Run this file from the repository root, for example::

    MPLBACKEND=Agg python validation/model.euler/euler-nozzle-convergence.py

It prints normalized L1 errors in Mach number and produces one figure for the
mesh study and one for the pseudo-time convergence of global and local CFL
steps.  Local time stepping is a steady-state acceleration: its horizontal
axis is iteration count, not physical time.
"""

import matplotlib.pyplot as plt
import numpy as np

import flowdyn.integration as integration
import flowdyn.mesh as mesh
import flowdyn.modeldisc as modeldisc
import flowdyn.modelphy.euler as euler
import flowdyn.solution.euler_nozzle as nozzle_solution
from flowdyn.xnum import muscl, vanleer


def section(x):
    """Return the area law; its throat is at x=5."""
    return 1.0 - 0.5 * np.exp(-0.5 * (x - 5.0) ** 2)


def normalized_l1(numerical, reference, variable="mach"):
    """Return the cell-average normalized L1 error for one physical variable."""
    num = numerical.phydata(variable)
    ref = reference.phydata(variable)
    return np.mean(np.abs(num - ref)) / np.mean(np.abs(ref))


def solve_nozzle(
    ncell, npr, *, integrator=integration.rk3ssp, dtlocal=False, monitors=None, endtime=100.0
):
    """Compute a steady nozzle solution and its Aerokit analytical reference."""
    curmesh = mesh.unimesh(ncell=ncell, length=10.0)
    model = euler.nozzle(sectionlaw=section)
    reference = nozzle_solution.nozzle(model, section(curmesh.centers()), NPR=npr).fdata(curmesh)
    bc_left = {"type": "insub", "ptot": npr, "rttot": 1.0}
    bc_right = {"type": "outsub", "p": 1.0}
    rhs = modeldisc.fvm(model, curmesh, muscl(vanleer), bcL=bc_left, bcR=bc_right)
    solver = integrator(curmesh, rhs, monitors=monitors)
    initial = rhs.fdata_fromprim([1.0, 0.1, 1.0])
    directives = {"dtlocal": True} if dtlocal else None
    solution = solver.solve(initial, 0.8, [endtime], directives=directives)[-1]
    return solution, reference, solver


def mesh_study(nprs=(1.05, 1.10), grids=(50, 100, 200)):
    """Compare grid convergence for a subsonic and a transonic nozzle."""
    errors = {npr: [] for npr in nprs}
    for npr in nprs:
        for ncell in grids:
            solution, reference, _ = solve_nozzle(ncell, npr)
            error = normalized_l1(solution, reference)
            errors[npr].append(error)
            print(f"NPR={npr:.2f}, cells={ncell:3d}, normalized L1(Mach)={error:.3e}")

    fig, axis = plt.subplots(figsize=(6, 4))
    for npr, values in errors.items():
        regime = "subsonic" if npr < 1.067 else "transonic (normal shock)"
        axis.loglog(grids, values, "o-", label=f"NPR={npr:.2f}: {regime}")
    axis.set(xlabel="number of cells", ylabel="normalized L1 error in Mach", title="Nozzle grid convergence")
    axis.grid(which="both", linestyle="--", color="0.7")
    axis.legend()
    return errors, fig


def time_step_study(npr=1.10, ncell=200):
    """Compare RK3/RK4 global and RK3 local pseudo-time stepping at fixed CFL."""
    histories = {}
    methods = (
        ("RK3 SSP, global time step", integration.rk3ssp, False),
        ("RK3 SSP, local time step", integration.rk3ssp, True),
        ("RK4, global time step", integration.rk4, False),
    )
    for label, integrator, dtlocal in methods:
        monitors = {"residual": {"name": "L2 residual", "frequency": 5}}
        _, _, solver = solve_nozzle(ncell, npr, integrator=integrator, dtlocal=dtlocal, monitors=monitors)
        residual = monitors["residual"]["output"]
        histories[label] = (residual._it, residual._value, solver.nit())
        print(f"{label}: {solver.nit()} pseudo-time iterations")

    fig, axis = plt.subplots(figsize=(6, 4))
    for label, (iterations, residuals, _) in histories.items():
        axis.semilogy(iterations, residuals, label=label)
    axis.set(xlabel="pseudo-time iteration", ylabel="L2 residual", title=f"Nozzle pseudo-time convergence (NPR={npr:.2f})")
    axis.grid(which="both", linestyle="--", color="0.7")
    axis.legend()
    return histories, fig


if __name__ == "__main__":
    mesh_study()
    time_step_study()
    plt.show()
