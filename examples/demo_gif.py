"""Short demo for the README and the website: the Panda arm on the fast circle (1.35 m/s).

NMPC (acados) next to operational-space control, on the 15 cm circle with a 0.7 s period
(experiment 1 of examples/circle_comparison.py). Under each arm, the joint speeds as a
percentage of the Panda's limits: above 100 % the real robot stops with an error.
Shown at quarter speed.

    python examples/demo_gif.py      (writes docs/figures/demo.gif; needs acados)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import casadi as ca  # noqa: E402

from anmpc import dynamics as dyn  # noqa: E402
from anmpc.baselines import OSC  # noqa: E402
from anmpc.nmpc import NMPC, NMPCConfig  # noqa: E402
from anmpc.sim import run  # noqa: E402
from anmpc.task import NO_FRICTION, Circle, Limits  # noqa: E402

INK, TEAL, ORANGE, MUTED, RED = "#0B1220", "#2a9d8f", "#FB923C", "#6B7280", "#C2410C"
PAPER, GRID = "#F7F7F5", "#E4E4DF"
PERIOD, DT = 0.7, 0.01
SLOW = 4               # quarter speed
FRAME_MS = 120


def experiment():
    task = Circle(PERIOD)
    dur = task.ramp + 2 * task.period
    nmpc = NMPC(task, NMPCConfig(name="anmpc_demo"))
    nmpc.set_friction_model(np.zeros(7), np.zeros(7))
    runs = {"nmpc": run(nmpc, task, dur, "NMPC (acados)", dt=DT, friction=NO_FRICTION),
            "osc": run(OSC(task), task, dur, "Operational-space control", dt=DT,
                       friction=NO_FRICTION)}
    for k, r in runs.items():
        print(k, r.summary())
    return task, runs


def chain(q):
    """World positions of the base, the seven joints and the TCP."""
    _, Rw, pw = dyn._transforms(ca.DM(q))
    pts = [np.zeros(3)] + [np.array(p).ravel() for p in pw]
    tcp = np.array(pw[-1]).ravel() + np.array(Rw[-1]) @ np.asarray(dyn.D.TCP_P)
    return np.array(pts + [tcp])


def draw(task, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    lim = Limits().dq_max
    cols = [(runs["nmpc"], TEAL, "NMPC (acados)"),
            (runs["osc"], ORANGE, "Operational-space control")]
    t = cols[0][0].t
    step = max(1, int(round(FRAME_MS / 1000 / SLOW / DT)))
    idx = list(range(0, len(t), step))
    if idx[-1] != len(t) - 1:
        idx.append(len(t) - 1)
    ref = cols[0][0].p_ref

    fig = plt.figure(figsize=(7.2, 4.6), dpi=100)
    fig.patch.set_facecolor(PAPER)
    arms = [fig.add_axes([0.0 + 0.5 * i, 0.34, 0.5, 0.58], projection="3d") for i in range(2)]
    bars = [fig.add_axes([0.08 + 0.5 * i, 0.09, 0.38, 0.19]) for i in range(2)]
    frames = []
    for k in idx:
        for ax3, axb, (r, color, label) in zip(arms, bars, cols):
            ax3.clear()
            ax3.set_facecolor(PAPER)
            P = chain(r.q[k])
            ax3.plot(ref[:, 0], ref[:, 1], ref[:, 2], color=MUTED, lw=0.8, ls=(0, (3, 3)))
            ax3.plot(r.p[: k + 1, 0], r.p[: k + 1, 1], r.p[: k + 1, 2], color=color, lw=1.6)
            ax3.plot(P[:, 0], P[:, 1], P[:, 2], "-o", color=INK, lw=3, ms=3.5)
            ax3.set(xlim=(-0.05, 0.65), ylim=(-0.35, 0.35), zlim=(0, 0.7))
            ax3.set_box_aspect((1, 1, 1), zoom=1.5)
            ax3.view_init(elev=18, azim=-60)
            ax3.set_axis_off()
            ax3.set_title(label, loc="left", fontsize=10, color=INK, fontweight="bold", y=0.98)

            ratio = 100 * np.abs(r.dq[k]) / lim
            peak = 100 * np.max(np.abs(r.dq[: k + 1]) / lim)
            axb.clear()
            axb.set_facecolor(PAPER)
            axb.bar(np.arange(1, 8), ratio, color=[RED if v > 101 else color for v in ratio],
                    width=0.7)
            axb.axhline(100, color=RED, lw=1, ls="--")
            axb.set(ylim=(0, 150), xticks=range(1, 8), yticks=(0, 50, 100, 150))
            axb.set_xlabel("joint", fontsize=8.5, color=MUTED, labelpad=1)
            axb.set_ylabel("speed, % of limit", fontsize=8.5, color=MUTED)
            axb.tick_params(labelsize=8, colors=MUTED)
            for s in ("top", "right"):
                axb.spines[s].set_visible(False)
            over = peak > 101
            axb.text(1.0, 1.04, f"peak so far {peak:.0f} %" + ("  (the robot would stop)" if over
                                                               else ""),
                     transform=axb.transAxes, ha="right", va="bottom", fontsize=8.5,
                     color=RED if over else INK)
        fig.suptitle(f"Panda arm, 15 cm circle at 1.35 m/s   ·   t = {t[k]:.2f} s "
                     f"(1/{SLOW} speed)", x=0.02, y=0.985, ha="left", fontsize=9.5,
                     color=MUTED)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    durations = [FRAME_MS] * len(frames)
    durations[-1] = 3000
    images = [Image.fromarray(fr).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
              for fr in frames]
    out = ROOT / "docs" / "figures" / "demo.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True)
    print(f"{len(frames)} frames; saved {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    draw(*experiment())
