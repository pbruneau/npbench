import numpy as np
import cupy as cp


def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 *
        dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) - ((u[1:-1, 2:] - u
        [1:-1, 0:-2]) / (2 * dx)) ** 2 - 2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) /
        (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) - ((v[2:, 1:-1
        ] - v[0:-2, 1:-1]) / (2 * dy)) ** 2)


def pressure_poisson(nit, p, dx, dy, b):
    pn = cp.empty_like(cp.asarray(p))
    pn = p.copy()
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 + (pn[2:,
            1:-1] + pn[0:-2, 1:-1]) * dx ** 2) / (2 * (dx ** 2 + dy ** 2)
            ) - dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]
        p[:, (-1)] = p[:, (-2)]
        p[(0), :] = p[(1), :]
        p[:, (0)] = p[:, (1)]
        p[(-1), :] = 0


def cavity_flow(nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu):
    if isinstance(nu, np.ndarray):
        nu = cp.asarray(nu)
    if isinstance(rho, np.ndarray):
        rho = cp.asarray(rho)
    if isinstance(p, np.ndarray):
        p = cp.asarray(p)
    if isinstance(dy, np.ndarray):
        dy = cp.asarray(dy)
    if isinstance(dx, np.ndarray):
        dx = cp.asarray(dx)
    if isinstance(dt, np.ndarray):
        dt = cp.asarray(dt)
    if isinstance(v, np.ndarray):
        v = cp.asarray(v)
    if isinstance(u, np.ndarray):
        u = cp.asarray(u)
    if isinstance(nit, np.ndarray):
        nit = cp.asarray(nit)
    if isinstance(nt, np.ndarray):
        nt = cp.asarray(nt)
    if isinstance(ny, np.ndarray):
        ny = cp.asarray(ny)
    if isinstance(nx, np.ndarray):
        nx = cp.asarray(nx)
    un = cp.empty_like(cp.asarray(u))
    vn = cp.empty_like(cp.asarray(v))
    b = cp.zeros((int(ny), int(nx)))
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        build_up_b(b, rho, dt, u, v, dx, dy)
        pressure_poisson(nit, p, dx, dy, b)
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-
            1, 1:-1] - un[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dy * (un[1:-
            1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) * (p[1:-1, 2:] -
            p[1:-1, 0:-2]) + nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:
            -1, 1:-1] + un[1:-1, 0:-2]) + dt / dy ** 2 * (un[2:, 1:-1] - 2 *
            un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-
            1, 1:-1] - vn[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dy * (vn[1:-
            1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) * (p[2:, 1:-1] -
            p[0:-2, 1:-1]) + nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:
            -1, 1:-1] + vn[1:-1, 0:-2]) + dt / dy ** 2 * (vn[2:, 1:-1] - 2 *
            vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        u[(0), :] = 0
        u[:, (0)] = 0
        u[:, (-1)] = 0
        u[(-1), :] = 1
        v[(0), :] = 0
        v[(-1), :] = 0
        v[:, (0)] = 0
        v[:, (-1)] = 0
