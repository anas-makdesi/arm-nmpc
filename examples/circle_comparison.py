"""Panda experiments: NMPC (acados) vs operational-space control and a task-space QP on a
fast Cartesian circle, and NMPC with a friction model learned from data with a certified
error bound (ddma).

    python examples/circle_comparison.py      # writes results/circle.json and figures

Experiment 1 - speed sweep (no joint friction in the simulated arm, so the comparison is
about dynamics and limits): circle of radius 15 cm, periods 2.0 s down to 0.6 s.
Experiment 2 - joint friction (viscous + Coulomb) in the simulated arm, unknown to the
controllers: NMPC without a friction model, with the model ddma certified from a 6 s
calibration motion, and with the true parameters (oracle); and the two baselines.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from anmpc.baselines import OSC, TaskQP  # noqa: E402
from anmpc.friction import FrictionCertificate, calibration_run  # noqa: E402
from anmpc.nmpc import NMPC, NMPCConfig  # noqa: E402
from anmpc.sim import run  # noqa: E402
from anmpc.task import NO_FRICTION, Circle, Friction  # noqa: E402

OUT = ROOT / "results"
FIG = ROOT / "docs" / "figures"
PERIODS = (2.0, 1.0, 0.8, 0.7, 0.6)
DT = 0.01


def main(fig: bool = True):
    OUT.mkdir(exist_ok=True)
    nmpc = NMPC(Circle(1.0), NMPCConfig(name="anmpc_main"))

    def fly_nmpc(task, friction, fv=None, fc=None, label="NMPC (acados)"):
        nmpc.task, nmpc.initialized, nmpc.solve_times, nmpc.status = task, False, [], []
        nmpc.set_friction_model(
            np.zeros(7) if fv is None else fv, np.zeros(7) if fc is None else fc
        )
        return run(nmpc, task, task.ramp + 2 * task.period, label, dt=DT, friction=friction)

    sweep, runs = {}, {}
    for T in PERIODS:
        task = Circle(T)
        r = {
            "nmpc": fly_nmpc(task, NO_FRICTION),
            "osc": run(
                OSC(task),
                task,
                task.ramp + 2 * T,
                "Operational-space control",
                dt=DT,
                friction=NO_FRICTION,
            ),
            "qp": run(
                TaskQP(task, dt=DT),
                task,
                task.ramp + 2 * T,
                "Task-space QP",
                dt=DT,
                friction=NO_FRICTION,
            ),
        }
        runs[T] = r
        sweep[T] = {k: v.summary() for k, v in r.items()}
        sweep[T]["peak_speed_m_s"] = round(task.peak_speed, 2)
        print(
            T,
            {
                k: (s["rms_error_mm"], s["max_dq_ratio"])
                for k, s in sweep[T].items()
                if isinstance(s, dict)
            },
        )

    # ------------------------------------------------------------ friction
    F = Friction()
    DQ, TF = calibration_run(F, amp=0.8)
    cert = FrictionCertificate()
    cert.add(DQ, TF)
    fv, fc, err = cert.fit()
    fr = {}
    for T in (2.0, 1.0, 0.7):
        task = Circle(T)
        fr[T] = {
            "none": fly_nmpc(task, F, label="NMPC, no friction model"),
            "certified": fly_nmpc(task, F, fv, fc, label="NMPC, ddma friction model"),
            "oracle": fly_nmpc(
                task, F, np.array(F.fv), np.array(F.fc), label="NMPC, true friction parameters"
            ),
            "osc": run(
                OSC(task), task, task.ramp + 2 * T, "Operational-space control", dt=DT, friction=F
            ),
            "qp": run(
                TaskQP(task, dt=DT), task, task.ramp + 2 * T, "Task-space QP", dt=DT, friction=F
            ),
        }
        print("friction", T, {k: v.summary()["rms_error_mm"] for k, v in fr[T].items()})
    summary = dict(
        sweep={str(T): s for T, s in sweep.items()},
        friction={str(T): {k: v.summary() for k, v in d.items()} for T, d in fr.items()},
        friction_model=dict(
            fv=fv.round(3).tolist(),
            fc=fc.round(3).tolist(),
            certified_error_Nm=err.round(3).tolist(),
            covered_velocity_rad_s=[[round(a, 2) for a in cert.coverage(i)] for i in range(7)],
            true_fv=list(F.fv),
            true_fc=list(F.fc),
            calibration_samples=int(len(DQ)),
            noise_Nm=cert.noise,
        ),
    )
    (OUT / "circle.json").write_text(json.dumps(summary, indent=2))
    if fig:
        figures(runs, fr, cert, DQ, TF, F, fv, fc)
    return summary


def figures(runs, fr, cert, DQ, TF, F, fv, fc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from anmpc.plotting import CRITICAL, INK2, MUTED, SERIES, style

    style()
    FIG.mkdir(parents=True, exist_ok=True)
    names = [
        ("nmpc", "NMPC (acados)"),
        ("qp", "Task-space QP"),
        ("osc", "Operational-space control"),
    ]

    # --- speed sweep: error and joint-velocity use
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    sp = [Circle(T).peak_speed for T in PERIODS]
    for i, (key, label) in enumerate(names):
        e = [runs[T][key].summary()["rms_error_mm"] for T in PERIODS]
        v = [100 * runs[T][key].summary()["max_dq_ratio"] for T in PERIODS]
        axs[0].plot(sp, e, "-o", color=SERIES[i], ms=6, mec="white", mew=1.2, label=label)
        axs[1].plot(sp, v, "-o", color=SERIES[i], ms=6, mec="white", mew=1.2, label=label)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("peak TCP speed on the circle (m/s)")
    axs[0].set_ylabel("RMS TCP error (mm)")
    axs[0].set_title("Tracking error (no joint friction)")
    axs[0].legend(loc="upper left", fontsize=9)
    axs[1].axhline(100, color=CRITICAL, lw=1.2, ls="--")
    axs[1].text(
        sp[0],
        102,
        "joint velocity limit (the robot stops above it)",
        color=INK2,
        fontsize=9,
        va="bottom",
    )
    axs[1].set_xlabel("peak TCP speed on the circle (m/s)")
    axs[1].set_ylabel("max joint speed (% of limit)")
    axs[1].set_title("Joint velocity limits")
    axs[1].set_ylim(40, 175)
    fig.tight_layout()
    fig.savefig(FIG / "circle_speed.png", dpi=150)
    plt.close(fig)

    # --- friction: certified band on joint 2 + errors
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axs[0]
    j = 1
    c, lo, hi = cert.band(j)
    ok = np.isfinite(lo) & np.isfinite(hi) & (c >= cert.v_lo[j]) & (c <= cert.v_hi[j])
    ax.scatter(
        DQ[:, j],
        TF[:, j],
        s=6,
        color=SERIES[0],
        alpha=0.35,
        lw=0,
        label="measured (applied - rigid-body torque)",
    )
    ax.fill_between(
        c[ok], lo[ok], hi[ok], color=SERIES[0], alpha=0.18, lw=0, label="ddma guaranteed band"
    )
    vv = np.linspace(cert.v_lo[j], cert.v_hi[j], 300)
    ax.plot(
        vv,
        fv[j] * vv + fc[j] * np.tanh(vv / cert.eps),
        color=SERIES[1],
        lw=2,
        label="smooth model used by the NMPC",
    )
    ax.plot(
        vv,
        F.fv[j] * vv + F.fc[j] * np.tanh(vv / F.eps),
        color=INK2,
        lw=1.2,
        ls="--",
        label="true friction (unknown)",
    )
    ax.set_xlabel("joint 2 velocity (rad/s)")
    ax.set_ylabel("friction torque (N m)")
    ax.set_title("Joint friction learned from a 6 s calibration motion")
    ax.legend(loc="upper left", fontsize=8.5)
    ax = axs[1]
    keys = [
        ("osc", "Operational-space control", MUTED),
        ("qp", "Task-space QP", "#d9d8d3"),
        ("none", "NMPC, no friction model", SERIES[2]),
        ("certified", "NMPC, ddma friction model", SERIES[0]),
        ("oracle", "NMPC, true friction (oracle)", SERIES[3]),
    ]
    Ts = list(fr.keys())
    w = 0.16
    for i, (k, label, col) in enumerate(keys):
        vals = [fr[T][k].summary()["rms_error_mm"] for T in Ts]
        x = np.arange(len(Ts)) + (i - 2) * w
        ax.bar(
            x,
            vals,
            width=w * 0.92,
            color=col,
            label=label,
            hatch="//" if k == "qp" else None,
            edgecolor=MUTED if k == "qp" else None,
            lw=0,
        )
    ax.set_yscale("log")
    ax.set_xticks(range(len(Ts)))
    ax.set_xticklabels([f"{Circle(T).peak_speed:.2f} m/s" for T in Ts])
    ax.set_ylabel("RMS TCP error (mm), log scale")
    ax.set_title("Tracking with unmodelled joint friction")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.set_ylim(0.02, 60)
    fig.tight_layout()
    fig.savefig(FIG / "friction.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(fig="--no-fig" not in sys.argv)
