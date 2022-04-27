###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################


from typing import List, Tuple

import numba
import numpy as np


@numba.njit
def bresenham_raycast(
    start_pt: np.ndarray, end_pts: np.ndarray, esdf: np.ndarray
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Raycast from `start_pt` to each `end_pt` in `end_pts`
    via the Bresenham Line Algorithm.

    Args:
        start_pt (np.ndarray): (2, ) array of the starting point.
        end_pts (np.ndarray): (N, 2) array of `N` end points.
        esdf (np.ndarray): (H, W) esdf.

    Returns:
        Tuple[List[List[int]], List[Tuple[int, int]]]:
            - List of ESDF values on the line between
              `start_pt` and each `end_pt` in `end_pts`.
            - List of ESDF coordinates used for each raycast.
    """
    ret_vals = []
    rays = []
    x0, y0 = start_pt
    for pt in end_pts:
        x1, y1 = pt
        v, r = bresenham_single_pt(x0, y0, x1, y1, esdf)
        ret_vals.append(v)
        rays.append(r)

    min_vals = []
    for v in ret_vals:
        min_vals.append(np.array(v).min())
    return min_vals, rays


@numba.njit
def line_high(
    x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray, esdf: np.ndarray
) -> Tuple[List[int], Tuple[int, int]]:
    """Perform raycast between negative sloped line
    (x0, y0), (x1, y1) using `esdf` data.

    Args:
        x0 (np.int): Start x point.
        y0 (np.int): Start y point.
        x1 (np.int): End x point.
        y0 (np.int): End y point.
        esdf (np.array): Shape (R, C) esdf.

    Returns:
        Tuple[List[float], List[Tuple[int, int]]]]
        - ESDF values from points along the raycast
    """
    values = []
    pixels = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    err = (2 * dx) - dy
    x = x0

    for y in np.arange(y0, y1 + 1):
        v = esdf[y, x]
        values.append(v)
        pixels.append((y, x))
        if err > 0:
            x = x + xi
            err += 2 * (dx - dy)
        else:
            err += 2 * dx
    return values, pixels


@numba.njit
def line_low(
    x0: np.int, y0: np.int, x1: np.int, y1: np.int, esdf: np.ndarray
) -> Tuple[List[int], Tuple[int, int]]:
    """Perform raycast between negative sloped line
    (x0, y0), (x1, y1) using `esdf` data.

    Args:
        x0 (np.int): Start x point.
        y0 (np.int): Start y point.
        x1 (np.int): End x point.
        y0 (np.int): End y point.
        esdf (np.array): Shape (R, C) esdf.

    Returns:
        Tuple[List[float], List[Tuple[int, int]]]]
        - ESDF values from points along the raycast
    """
    values = []
    pixels = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    err = (2 * dy) - dx
    y = y0

    for x in np.arange(x0, x1 + 1):
        v = esdf[y, x]
        values.append(v)
        pixels.append((y, x))

        if err > 0:
            y = y + yi
            err += 2 * (dy - dx)
        else:
            err += 2 * dy
    return values, pixels


@numba.njit
def bresenham_single_pt(
    x0: np.int, y0: np.int, x1: np.int, y1: np.int, esdf: np.ndarray
) -> Tuple[List[int], Tuple[int, int]]:
    """Perform raycast between point (x0, y0) and
    (x1, y1) using `esdf` data.

    Args:
        x0 (np.int): Start x point.
        y0 (np.int): Start y point.
        x1 (np.int): End x point.
        y0 (np.int): End y point.
        esdf (np.array): Shape (R, C) esdf.

    Returns:
        Tuple[List[float], List[Tuple[int, int]]]]
        - ESDF values from points along the raycast
    """
    if np.abs(y1 - y0) < np.abs(x1 - x0):
        if x0 > x1:
            v, p = line_low(x1, y1, x0, y0, esdf)
        else:
            v, p = line_low(x0, y0, x1, y1, esdf)
    else:
        if y0 > y1:
            v, p = line_high(x1, y1, x0, y0, esdf)
        else:
            v, p = line_high(x0, y0, x1, y1, esdf)
    return v, p
