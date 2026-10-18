"""Validate quasi-one-dimensional nozzle computations against Aerokit.

The case uses the smooth converging-diverging area law from the nozzle
examples.  The reference Mach number is computed directly with
``aerokit.instance.nozzle.nozzle``, while Flowdyn evolves the finite-volume
discretization to a steady state.

Two pressure ratios are deliberately retained:

* ``NPR=1.05`` is wholly subsonic, so the reference contains no discontinuity;
* ``NPR=1.10`` is choked with a normal shock in the divergent part.  Its
  discontinuity limits the global order, which is expected rather than a
  regression.

Run this file from the repository root, for example::

    MPLBACKEND=Agg python validation/model.nozzle/euler-nozzle-convergence.py

It prints normalized L1 errors in Mach number and produces separate figures
for Mach, pressures, total-pressure/mass-flow defects, and pseudo-time
convergence.  Local time stepping is a steady-state acceleration: its
horizontal axis is iteration count, not physical time.
"""

import matplotlib.pyplot as plt
import numpy as np

from aerokit.instance.nozzle import nozzle as aerokit_nozzle
import flowdyn.integration as integration
import flowdyn.mesh as mesh
import flowdyn.modeldisc as modeldisc
import flowdyn.modelphy.euler as euler
from flowdyn.xnum import muscl, vanleer


def section(x):
    """Return the area law; its throat is at x=5."""
    return 1.0 - 0.5 * np.exp(-0.5 * (x - 5.0) ** 2)


def throat_refined_mesh(ncell, sharpness=3.0):
    """Return a symmetric mesh concentrated at the throat x=5.

    The hyperbolic-sine mapping is monotonic and maps both endpoints exactly.
    With the default sharpness, the cells near the throat are about ten times
    smaller than at the inlet and outlet.  This deliberately exposes the
    difference between global and local pseudo-time steps.
    """

    def cluster_at_throat(x):
        coordinate = (x - 5.0) / 5.0
        return 5.0 + 5.0 * np.sinh(sharpness * coordinate) / np.sinh(sharpness)

    return mesh.morphedmesh(ncell=ncell, length=10.0, morph=cluster_at_throat)


def aerokit_mach(curmesh, gamma, npr):
    """Compute the analytical Mach distribution with ``aerokit.instance.nozzle``."""
    return aerokit_nozzle(
        x=curmesh.centers(), section=section(curmesh.centers()), gamma=gamma, NPR=npr
    ).Mach()


def normalized_l1(numerical, reference_mach):
    """Return the cell-average normalized L1 error in Mach number."""
    num = numerical.phydata("mach")
    return np.mean(np.abs(num - reference_mach)) / np.mean(np.abs(reference_mach))


def solve_nozzle(
    ncell,
    npr,
    *,
    integrator=integration.rk3ssp,
    cfl=0.8,
    dtlocal=False,
    monitors=None,
    endtime=500.0,
    maxit=None,
    refine_throat=True,
):
    """Compute a steady nozzle solution and its Aerokit Mach reference."""
    curmesh = throat_refined_mesh(ncell) if refine_throat else mesh.unimesh(ncell=ncell, length=10.0)
    model = euler.nozzle(sectionlaw=section)
    reference_mach = aerokit_mach(curmesh, model.gamma, npr)
    bc_left = {"type": "insub", "ptot": npr, "rttot": 1.0}
    bc_right = {"type": "outsub", "p": 1.0}
    rhs = modeldisc.fvm(model, curmesh, muscl(vanleer), bcL=bc_left, bcR=bc_right)
    solver = integrator(curmesh, rhs, monitors=monitors)
    initial = rhs.fdata_fromprim([1.0, 0.1, 1.0])
    directives = {"dtlocal": True} if dtlocal else None
    if maxit is None:
        solution = solver.solve(initial, cfl, [endtime], directives=directives)[-1]
    else:
        solution = solver.solve(initial, cfl, stop={"maxit": maxit}, directives=directives)[-1]
    return solution, reference_mach, solver


def solve_straight_duct(ncell, npr, *, endtime=500.0):
    """Compute the matching straight-duct case, which has no geometry source."""
    curmesh = mesh.unimesh(ncell=ncell, length=10.0)
    model = euler.model()
    bc_left = {"type": "insub", "ptot": npr, "rttot": 1.0}
    bc_right = {"type": "outsub", "p": 1.0}
    rhs = modeldisc.fvm(model, curmesh, muscl(vanleer), bcL=bc_left, bcR=bc_right)
    solver = integration.rk3ssp(curmesh, rhs)
    initial = rhs.fdata_fromprim([1.0, 0.1, 1.0])
    return solver.solve(initial, 0.8, [endtime])[-1]


def total_pressure_metrics(solution):
    """Measure artificial total-pressure and mass-flow defects in a field.

    In the smooth subsonic nozzle, inviscid theory predicts constant total
    pressure and mass flow.  The inlet-to-outlet loss detects remaining
    pseudo-time error, whereas the maximum internal deficit identifies the
    flux/geometry-source imbalance.
    """
    ptot = solution.phydata("ptot")
    massflow = solution.phydata("massflow")
    ptot_inlet = ptot[0]
    massflow_inlet = massflow[0]
    return {
        "outlet_loss": 100.0 * (ptot_inlet - ptot[-1]) / ptot_inlet,
        "internal_deficit": 100.0 * (ptot_inlet - np.min(ptot)) / ptot_inlet,
        "massflow_spread": 100.0 * np.ptp(massflow) / massflow_inlet,
    }


def mesh_study(nprs=(1.05, 1.10), grids=(50, 100, 200)):
    """Compare grid convergence for a subsonic and a transonic nozzle."""
    errors = {npr: [] for npr in nprs}
    for npr in nprs:
        for ncell in grids:
            # Use uniform grids to isolate spatial mesh refinement from the
            # separate local-time-step demonstration below.
            solution, reference_mach, _ = solve_nozzle(ncell, npr, refine_throat=False)
            error = normalized_l1(solution, reference_mach)
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


def mach_comparison(nprs=(1.05, 1.10), ncell=200):
    """Plot Flowdyn and direct Aerokit Mach distributions for both regimes."""
    fig, axes = plt.subplots(1, len(nprs), figsize=(11, 4), sharey=False)
    axes = np.atleast_1d(axes)
    for axis, npr in zip(axes, nprs):
        solution, reference_mach, _ = solve_nozzle(ncell, npr, refine_throat=False)
        xcoord = solution.mesh.centers()
        regime = "subsonic" if npr < 1.067 else "transonic, normal shock"
        axis.plot(xcoord, solution.phydata("mach"), "o", ms=3, label="Flowdyn")
        axis.plot(xcoord, reference_mach, "k-", lw=2, label="Aerokit instance.nozzle")
        axis.axvline(5.0, color="0.5", linestyle="--", label="throat")
        axis.set(xlabel="x", ylabel="Mach number", title=f"NPR={npr:.2f}: {regime}")
        axis.grid(linestyle="--", color="0.7")
        axis.legend()
    fig.suptitle("Nozzle Mach: Flowdyn versus Aerokit")
    fig.tight_layout()
    return fig


def pressure_comparison(nprs=(1.05, 1.10), ncell=200):
    """Plot static and total pressure from Flowdyn and Aerokit for both NPRs.

    ``aerokit.instance.nozzle`` uses an outlet static-pressure reference of
    ``1/NPR``.  Multiplication by ``NPR`` maps its pressures to the Flowdyn
    boundary-condition scale, for which the outlet static pressure is one.
    """
    fig, axes = plt.subplots(len(nprs), 2, figsize=(11, 7), sharex=True, squeeze=False)
    for row, npr in enumerate(nprs):
        solution, _, _ = solve_nozzle(ncell, npr, refine_throat=False)
        xcoord = solution.mesh.centers()
        reference = aerokit_nozzle(
            x=xcoord, section=section(xcoord), gamma=solution.model.gamma, NPR=npr
        )
        pressures = (
            ("pressure", reference.Ps() * npr, r"$p_s$"),
            ("ptot", reference.Ptot() * npr, r"$p_t$"),
        )
        for axis, (name, aerokit_pressure, symbol) in zip(axes[row], pressures):
            axis.plot(xcoord, solution.phydata(name), "o", ms=3, label="Flowdyn")
            axis.plot(xcoord, aerokit_pressure, "k-", lw=2, label="Aerokit instance.nozzle")
            axis.axvline(5.0, color="0.5", linestyle="--", label="throat")
            axis.set(ylabel=symbol, title=f"NPR={npr:.2f}: {symbol}(x)")
            axis.grid(linestyle="--", color="0.7")
            axis.legend()
    for axis in axes[-1]:
        axis.set_xlabel("x")
    fig.suptitle("Nozzle pressures: Flowdyn versus Aerokit")
    fig.tight_layout()
    return fig


def total_pressure_loss_study(npr=1.05, grids=(50, 100, 200), endtime=1000.0):
    """Quantify subsonic nozzle losses and compare them with a straight duct.

    The straight duct has the same inlet/outlet boundary conditions, flux, and
    time integrator as the nozzle.  Its only difference is the absence of the
    area source term.  It is therefore a control case for identifying a
    geometry-source/discretization effect.
    """
    cases = {"nozzle": [], "straight duct": []}
    for ncell in grids:
        nozzle_solution, _, _ = solve_nozzle(ncell, npr, refine_throat=False, endtime=endtime)
        duct_solution = solve_straight_duct(ncell, npr, endtime=endtime)
        for label, solution in (("nozzle", nozzle_solution), ("straight duct", duct_solution)):
            metric = total_pressure_metrics(solution)
            cases[label].append(metric)
            print(
                f"{label:13s}, cells={ncell:3d}, "
                f"ptot outlet loss={metric['outlet_loss']:.3e}%, "
                f"ptot internal deficit={metric['internal_deficit']:.3e}%, "
                f"mass-flow spread={metric['massflow_spread']:.3e}%"
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, metrics in cases.items():
        axes[0].loglog(grids, [value["internal_deficit"] for value in metrics], "o-", label=label)
        axes[1].loglog(grids, [value["massflow_spread"] for value in metrics], "o-", label=label)
    axes[0].set(
        xlabel="number of cells",
        ylabel=r"max $(p_{t,in}-p_t)/p_{t,in}$ (%)",
        title=f"Subsonic total-pressure defect (NPR={npr:.2f})",
    )
    axes[1].set(
        xlabel="number of cells",
        ylabel=r"max $(\dot m)-\min(\dot m)$ over $\dot m_{in}$ (%)",
        title="Mass-flow conservation defect",
    )
    for axis in axes:
        axis.grid(which="both", linestyle="--", color="0.7")
        axis.legend()
    fig.tight_layout()
    return cases, fig


def effective_cfl(solver, local_time_step):
    """Return the cell CFL numbers used in the last pseudo-time iteration."""
    local_dt = solver.modeldisc.calc_timestep(solver.Qn, solver.condition)
    if local_time_step:
        return np.full_like(local_dt, solver.condition)
    return solver.condition * np.min(local_dt) / local_dt


def time_step_study(npr=1.10, ncell=200, maxit=10000):
    """Compare RK3/RK4 global and RK3 local pseudo-time stepping at fixed CFL."""
    histories = {}
    methods = (
        ("RK3 SSP, global time step", integration.rk3ssp, 0.8, False),
        ("RK3 SSP, local time step", integration.rk3ssp, 0.8, True),
        ("RK4, global time step", integration.rk4, 1.0, False),
    )
    for label, integrator, cfl, dtlocal in methods:
        monitors = {"residual": {"name": "L2 residual", "frequency": 5}}
        _, _, solver = solve_nozzle(
            ncell, npr, integrator=integrator, cfl=cfl, dtlocal=dtlocal, monitors=monitors, maxit=maxit
        )
        residual = monitors["residual"]["output"]
        cfl_cells = effective_cfl(solver, dtlocal)
        histories[label] = (residual._it, residual._value, solver.nit(), cfl_cells)
        print(
            f"{label}: requested CFL={cfl:.2f}, "
            f"effective cell CFL=[{cfl_cells.min():.3f}, {cfl_cells.max():.3f}], "
            f"{solver.nit()} pseudo-time iterations"
        )

    curmesh = throat_refined_mesh(ncell)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(curmesh.centers(), curmesh.dx(), "k.-")
    axes[0].axvline(5.0, color="0.5", linestyle="--", label="throat")
    axes[0].set(xlabel="x", ylabel="cell width", title="Throat-refined mesh")
    axes[0].grid(linestyle="--", color="0.7")
    axes[0].legend()
    for label, (iterations, residuals, _, cfl_cells) in histories.items():
        axes[1].semilogy(iterations, residuals, label=label)
        axes[2].plot(curmesh.centers(), cfl_cells, label=label)
    axes[1].set(xlabel="pseudo-time iteration", ylabel="L2 residual", title=f"Convergence (NPR={npr:.2f})")
    axes[2].set(xlabel="x", ylabel="effective cell CFL", title="CFL used by each method")
    for axis in axes[1:]:
        axis.grid(which="both", linestyle="--", color="0.7")
        axis.legend()
    return histories, fig


if __name__ == "__main__":
    mesh_study()
    mach_comparison()
    pressure_comparison()
    total_pressure_loss_study()
    time_step_study()
    if "agg" not in plt.get_backend().lower():
        plt.show()
