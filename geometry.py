"""Geometry shared by the command-line renderer and Tk GUI.

Viewport intersections are display points, not LP vertices. Equalities remain
equalities when clipping, so segments and isolated points are preserved.
"""
from itertools import combinations
import numpy as np
from exact_engine import number


def linear_solve(A, b):
    n = len(b)
    rows = [list(a) + [v] for a, v in zip(A, b)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if rows[i][j]), None)
        if pivot is None:
            return None
        rows[j], rows[pivot] = rows[pivot], rows[j]
        v = rows[j][j]
        rows[j] = [a/v for a in rows[j]]
        for i in range(n):
            if i != j:
                v = rows[i][j]
                rows[i] = [a-v*c for a, c in zip(rows[i], rows[j])]
    return tuple(row[-1] for row in rows)


def vertices(A, b, signs, n, nonnegative=True, bounds=None):
    rows = [[number(v) for v in row] for row in A]
    rhs = [number(v) for v in b]
    signs = list(signs)
    for j in range(n):
        e = [number(int(k == j)) for k in range(n)]
        if nonnegative:
            rows.append(e); rhs.append(number(0)); signs.append('>=')
        if bounds is not None:
            rows.extend([e, e]); rhs.extend(map(number, bounds[j])); signs.extend(['>=', '<='])
    result = set()
    for combo in combinations(range(len(rows)), n):
        x = linear_solve([rows[i] for i in combo], [rhs[i] for i in combo])
        if x is None:
            continue
        feasible = True
        for row, v, sign in zip(rows, rhs, signs):
            delta = sum(a*xx for a, xx in zip(row, x))-v
            if (sign == '<=' and delta > 0) or (sign == '>=' and delta < 0) or (sign == '=' and delta != 0):
                feasible = False
                break
        if feasible:
            result.add(x)
    return np.array(sorted(result), dtype=float).reshape(-1, n)


def draw_region(ax, points, alpha=.2, color='cyan'):
    """Render full, planar, linear and point regions without a Qhull crash."""
    if not len(points):
        return
    n = points.shape[1]
    center = points.mean(axis=0)
    _, singular, vh = np.linalg.svd(points-center, full_matrices=False)
    rank = np.linalg.matrix_rank(points-center)
    if rank == 1:
        order = np.argsort((points-center) @ vh[0])
        ax.plot(*points[order].T, color='navy', linewidth=3)
    elif rank >= 2:
        if rank == 2:
            uv = (points-center) @ vh[:2].T
            order = np.argsort(np.arctan2(uv[:, 1], uv[:, 0]))
            faces = [points[order]]
        else:
            from scipy.spatial import ConvexHull
            faces = [points[ids] for ids in ConvexHull(points).simplices]
        if n == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='navy'))
        else:
            from matplotlib.patches import Polygon
            ax.add_patch(Polygon(faces[0], alpha=alpha, facecolor=color, edgecolor='navy'))
    ax.scatter(*points.T, color='navy', s=20)
