"""Rigid-body dynamics of the 7-joint Panda arm in CasADi (Featherstone's spatial algebra).

* forward kinematics of the tool centre point (TCP),
* RNEA - inverse dynamics tau = M(q) qdd + C(q, dq) dq + g(q),
* ABA  - forward dynamics qdd = M(q)^-1 (tau - C dq - g), O(n), no matrix inverse.

Written once symbolically; the numeric functions (plant, baselines) are CasADi Functions of
the same expressions. Tested against Pinocchio (tests/test_dynamics.py).

Conventions: spatial motion vectors are [angular; linear]; each joint frame i is placed in
its parent by (JOINT_R, JOINT_P) and rotates about its own z axis by q_i.
"""

from __future__ import annotations

from functools import lru_cache

import casadi as ca
import numpy as np

from . import panda_data as D

NJ = 7


def _skew(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]), ca.horzcat(v[2], 0, -v[0]), ca.horzcat(-v[1], v[0], 0)
    )


def _rotz(q):
    c, s = ca.cos(q), ca.sin(q)
    return ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))


def _X(E, r):
    """Motion transform to a frame with rotation E (parent->child coords) at r (parent)."""
    Z = ca.SX.zeros(3, 3)
    return ca.vertcat(ca.horzcat(E, Z), ca.horzcat(-ca.mtimes(E, _skew(r)), E))


def _crm(v):
    w, u = v[0:3], v[3:6]
    Z = ca.SX.zeros(3, 3)
    return ca.vertcat(ca.horzcat(_skew(w), Z), ca.horzcat(_skew(u), _skew(w)))


def _crf(v):
    return -_crm(v).T


def _inertia(i):
    m, c, Ic = D.MASS[i], D.COM[i], D.I_COM[i]
    C = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    return ca.DM(np.block([[Ic + m * C @ C.T, m * C], [m * C.T, m * np.eye(3)]]))


S = ca.DM([0, 0, 1, 0, 0, 0])


def _transforms(q):
    """Child-from-parent motion transforms and the joints' poses in the world."""
    Xs, Rw, pw = [], [], []
    R, p = ca.DM.eye(3), ca.DM.zeros(3)
    for i in range(NJ):
        Ri = ca.mtimes(ca.DM(D.JOINT_R[i]), _rotz(q[i]))  # child orientation in the parent
        pi = ca.DM(D.JOINT_P[i])
        Xs.append(_X(Ri.T, pi))
        p = p + ca.mtimes(R, pi)
        R = ca.mtimes(R, Ri)
        Rw.append(R)
        pw.append(p)
    return Xs, Rw, pw


def fk_expr(q):
    """TCP position (3) and rotation (3x3) in the world (robot base) frame."""
    _, Rw, pw = _transforms(q)
    p = pw[-1] + ca.mtimes(Rw[-1], ca.DM(D.TCP_P))
    R = ca.mtimes(Rw[-1], ca.DM(D.TCP_R))
    return p, R


def rnea_expr(q, dq, ddq, gravity=True):
    Xs, _, _ = _transforms(q)
    a0 = ca.vertcat(0, 0, 0, -ca.DM(D.GRAVITY)) if gravity else ca.SX.zeros(6)
    v, a, f = [], [], []
    for i in range(NJ):
        vJ = S * dq[i]
        vp = ca.SX.zeros(6) if i == 0 else v[i - 1]
        ap = a0 if i == 0 else a[i - 1]
        vi = ca.mtimes(Xs[i], vp) + vJ
        ai = ca.mtimes(Xs[i], ap) + S * ddq[i] + ca.mtimes(_crm(vi), vJ)
        Ii = _inertia(i)
        v.append(vi)
        a.append(ai)
        f.append(ca.mtimes(Ii, ai) + ca.mtimes(_crf(vi), ca.mtimes(Ii, vi)))
    tau = [None] * NJ
    for i in range(NJ - 1, -1, -1):
        tau[i] = ca.mtimes(S.T, f[i])
        if i > 0:
            f[i - 1] = f[i - 1] + ca.mtimes(Xs[i].T, f[i])
    return ca.vertcat(*tau)


def aba_expr(q, dq, tau):
    Xs, _, _ = _transforms(q)
    a0 = ca.vertcat(0, 0, 0, -ca.DM(D.GRAVITY))
    v, c, IA, pA = [], [], [], []
    for i in range(NJ):
        vJ = S * dq[i]
        vp = ca.SX.zeros(6) if i == 0 else v[i - 1]
        vi = ca.mtimes(Xs[i], vp) + vJ
        Ii = _inertia(i)
        v.append(vi)
        c.append(ca.mtimes(_crm(vi), vJ))
        IA.append(Ii)
        pA.append(ca.mtimes(_crf(vi), ca.mtimes(Ii, vi)))
    U, Dd, u = [None] * NJ, [None] * NJ, [None] * NJ
    for i in range(NJ - 1, -1, -1):
        U[i] = ca.mtimes(IA[i], S)
        Dd[i] = ca.mtimes(S.T, U[i])
        u[i] = tau[i] - ca.mtimes(S.T, pA[i])
        if i > 0:
            Ia = IA[i] - ca.mtimes(U[i], U[i].T) / Dd[i]
            pa = pA[i] + ca.mtimes(Ia, c[i]) + U[i] * u[i] / Dd[i]
            IA[i - 1] = IA[i - 1] + ca.mtimes(Xs[i].T, ca.mtimes(Ia, Xs[i]))
            pA[i - 1] = pA[i - 1] + ca.mtimes(Xs[i].T, pa)
    qdd = [None] * NJ
    a = None
    for i in range(NJ):
        ap = a0 if i == 0 else a
        a_ = ca.mtimes(Xs[i], ap) + c[i]
        qdd[i] = (u[i] - ca.mtimes(U[i].T, a_)) / Dd[i]
        a = a_ + S * qdd[i]
    return ca.vertcat(*qdd)


# ------------------------------------------------------------------ numeric functions
@lru_cache(maxsize=None)
def functions():
    q, dq, x = ca.SX.sym("q", NJ), ca.SX.sym("dq", NJ), ca.SX.sym("x", NJ)
    p, R = fk_expr(q)
    Jp = ca.jacobian(p, q)
    return dict(
        fk=ca.Function("fk", [q], [p, R]),
        jac=ca.Function("tcp_jacobian", [q], [Jp]),
        jdot_dq=ca.Function("jdot_dq", [q, dq], [ca.mtimes(ca.jacobian(ca.mtimes(Jp, dq), q), dq)]),
        rnea=ca.Function("rnea", [q, dq, x], [rnea_expr(q, dq, x)]),
        aba=ca.Function("aba", [q, dq, x], [aba_expr(q, dq, x)]),
        mass=ca.Function(
            "mass",
            [q],
            [
                ca.substitute(
                    ca.jacobian(rnea_expr(q, ca.SX.zeros(NJ), x, False), x), x, ca.SX.zeros(NJ)
                )
            ],
        ),
    )


def fk(q):
    p, R = functions()["fk"](q)
    return np.array(p).ravel(), np.array(R)


def jacobian(q):
    return np.array(functions()["jac"](q))


def rnea(q, dq, ddq):
    return np.array(functions()["rnea"](q, dq, ddq)).ravel()


def aba(q, dq, tau):
    return np.array(functions()["aba"](q, dq, tau)).ravel()


def mass_matrix(q):
    return np.array(functions()["mass"](q))
