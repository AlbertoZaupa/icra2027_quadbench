#!/usr/bin/env python3
"""Generate the acados external functions once during the CMake build."""

import argparse
import os
from pathlib import Path

import casadi as ca


def quat_mul(a, b):
    return ca.vertcat(
        a[0]*b[0] - ca.dot(a[1:4], b[1:4]),
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    )


def rotation(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)),
        ca.horzcat(2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)),
        ca.horzcat(2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    nx, nu, np_ = 13, 4, 4
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    p = ca.SX.sym("p", np_)
    q = x[3:7]
    velocity = x[7:10]
    omega = x[10:13]

    mass, gravity = 0.75, 9.81
    inertia = ca.DM([0.0025, 0.0021, 0.0043])
    drag = ca.DM([0.26, 0.28, 0.42])
    horizontal_drag = 0.018
    arm = 0.14
    beta = 56.0 * ca.pi / 180.0
    kappa = 2.37e-8 / 1.51e-6

    rot = rotation(q)
    vb = rot.T @ velocity
    f_body = ca.vertcat(-drag[0]*vb[0], -drag[1]*vb[1],
                        -drag[2]*vb[2] + horizontal_drag*(vb[0]**2 + vb[1]**2))
    thrust = ca.sum1(u)
    acceleration = rot @ (f_body + ca.vertcat(0, 0, thrust))/mass + ca.DM([0, 0, -gravity])
    qdot = 0.5 * quat_mul(q, ca.vertcat(0, omega))

    lx, ly = arm*ca.sin(beta), arm*ca.cos(beta)
    torque = ca.vertcat(
        lx*(u[0]-u[1]-u[2]+u[3]),
        ly*(-u[0]-u[1]+u[2]+u[3]),
        kappa*(u[0]-u[1]+u[2]-u[3]),
    )
    iw = inertia*omega
    omega_dot = (torque - ca.cross(omega, iw))/inertia
    f_expl = ca.vertcat(velocity, qdot, acceleration, omega_dot)

    sx = ca.SX.sym("Sx", nx, nx)
    su = ca.SX.sym("Su", nx, nu)
    lambda_x = ca.SX.sym("lambdaX", nx)
    vde_x = ca.jtimes(f_expl, x, sx)
    vde_u = ca.jacobian(f_expl, u) + ca.jtimes(f_expl, x, su)
    adj = ca.jtimes(f_expl, ca.vertcat(x, u), lambda_x, True)

    expl_fun = ca.Function("quadrotor_expl_ode_fun", [x, u, p], [f_expl], {"cse": True})
    expl_forw = ca.Function("quadrotor_expl_vde_forw", [x, sx, su, u, p],
                            [f_expl, vde_x, vde_u], {"cse": True})
    expl_adj = ca.Function("quadrotor_expl_vde_adj", [x, lambda_x, u, p], [adj], {"cse": True})

    z = ca.SX.sym("z", 0)
    time = ca.SX.sym("t")
    q_ref = p
    q_ref_inv = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
    q_error = quat_mul(q, q_ref_inv)[1:4]
    y = ca.vertcat(x[0:3], q_error, x[7:10], x[10:13], u)
    ux = ca.vertcat(u, x)
    jac_t = ca.jacobian(y, ux).T
    dz = ca.SX.zeros(y.numel(), 0)
    cost_fun = ca.Function("quadrotor_cost_y_fun", [x, u, z, time, p], [y], {"cse": True})
    cost_jac = ca.Function("quadrotor_cost_y_fun_jac_ut_xt", [x, u, z, time, p],
                           [y, jac_t, dz], {"cse": True})

    u_e = ca.SX.sym("u_e", 0)
    z_e = ca.SX.sym("z_e", 0)
    y_e = ca.vertcat(x[0:3], q_error, x[7:10], x[10:13])
    jac_e_t = ca.jacobian(y_e, ca.vertcat(u_e, x)).T
    dz_e = ca.SX.zeros(y_e.numel(), 0)
    cost_e_fun = ca.Function("quadrotor_cost_y_e_fun", [x, u_e, z_e, time, p], [y_e], {"cse": True})
    cost_e_jac = ca.Function("quadrotor_cost_y_e_fun_jac_ut_xt", [x, u_e, z_e, time, p],
                             [y_e, jac_e_t, dz_e], {"cse": True})

    old_cwd = os.getcwd()
    try:
        os.chdir(output)
        generator = ca.CodeGenerator("quadrotor_generated.c", {
            "with_header": True,
            "with_mem": False,
            "mex": False,
            # acados' external-function ABI uses 32-bit workspace and
            # sparsity indices, independently of the CasADi wheel default.
            "casadi_int": "int",
        })
        for fun in (expl_fun, expl_forw, expl_adj, cost_fun, cost_jac, cost_e_fun, cost_e_jac):
            generator.add(fun)
        generator.generate()
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
