import numpy as np
import cupy as cp


def hdiff(in_field, out_field, coeff):
    if isinstance(coeff, np.ndarray):
        coeff = cp.asarray(coeff)
    if isinstance(out_field, np.ndarray):
        out_field = cp.asarray(out_field)
    if isinstance(in_field, np.ndarray):
        in_field = cp.asarray(in_field)
    I, J, K = out_field.shape[0], out_field.shape[1], out_field.shape[2]
    lap_field = 4.0 * in_field[1:I + 3, 1:J + 3, :] - (in_field[2:I + 4, 1:
        J + 3, :] + in_field[0:I + 2, 1:J + 3, :] + in_field[1:I + 3, 2:J +
        4, :] + in_field[1:I + 3, 0:J + 2, :])
    res = lap_field[1:, 1:J + 1, :] - lap_field[:-1, 1:J + 1, :]
    flx_field = cp.where(res * (in_field[2:I + 3, 2:J + 2, :] - in_field[1:
        I + 2, 2:J + 2, :]) > 0, 0, cp.asarray(res))
    res = lap_field[1:I + 1, 1:, :] - lap_field[1:I + 1, :-1, :]
    fly_field = cp.where(res * (in_field[2:I + 2, 2:J + 3, :] - in_field[2:
        I + 2, 1:J + 2, :]) > 0, 0, cp.asarray(res))
    out_field[:, :, :] = in_field[2:I + 2, 2:J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :] - flx_field[:-1, :, :] + fly_field[:, 1:, :] -
        fly_field[:, :-1, :])
