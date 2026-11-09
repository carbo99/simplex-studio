"""Test di regressione: SimplexSolver vs SciPy (stato, valore ottimo, dualità forte)."""
import os, random
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from scipy.optimize import linprog
from simplex import SimplexSolver


def _scipy(c, A, b, signs, is_min):
    Aub, bub, Aeq, beq = [], [], [], []
    for row, bi, s in zip(A, b, signs):
        if s == '<=': Aub.append(row); bub.append(bi)
        elif s == '>=': Aub.append([-v for v in row]); bub.append(-bi)
        else: Aeq.append(row); beq.append(bi)
    obj = c if is_min else [-v for v in c]
    kw = dict(A_ub=Aub or None, b_ub=bub or None, A_eq=Aeq or None, b_eq=beq or None, method='highs')
    res = linprog(obj, **kw)
    if res.status == 2:
        # Con il presolve HiGHS confonde "infeasible" e "unbounded": ricontrolla senza presolve
        alt = linprog(obj, options={'presolve': False}, **kw)
        if alt.status in (0, 3):
            res = alt
    return res


def check(c, A, b, signs, is_min):
    cc = [-x for x in c] if is_min else c
    s = SimplexSolver(cc, A, b, signs=signs, is_min=is_min)
    r = s.solve(verbose=False, plot3d=False)
    ref = _scipy(c, A, b, signs, is_min)
    ref_status = {0: 'optimal', 2: 'infeasible', 3: 'unbounded'}.get(ref.status, 'other')
    assert s.status == ref_status, (s.status, ref_status, c, A, b, signs, is_min)
    if s.status == 'optimal':
        z = -r[1] if is_min else r[1]
        z_ref = ref.fun if is_min else -ref.fun
        assert abs(z - z_ref) < 1e-6, (z, z_ref)
        y = s.get_dual_solution()
        assert abs(float(np.dot(b, y)) - z) < 1e-6, ("dualita forte", y, z)
    return s.status


def test_examples():
    check([3, 2, 4], [[1, 1, 1], [2, 1, 3], [1, 2, 1]], [4, 6, 5], ['<='] * 3, False)
    check([1, 3], [[6, 10], [3, 2], [1, -2], [0, 1]], [30, 6, -1, .5], ['<=', '>=', '>=', '>='], False)
    check([2, 3], [[1, 1], [1, 3]], [4, 6], ['>=', '>='], True)          # min: segno duale
    check([1, 1], [[1, 2], [1, 1]], [4, 3], ['=', '<='], False)           # uguaglianza
    check([1, 2], [[1, 1], [2, 2], [1, 0]], [2, 4, 1.5], ['=', '=', '<='], False)  # ridondante
    assert check([1, 1], [[1, -1]], [1], ['<='], False) == 'unbounded'
    assert check([1, 1], [[1, 1], [1, 1]], [1, 3], ['<=', '>='], False) == 'infeasible'


def test_random(n_cases=1500, seed=1):
    random.seed(seed); np.random.seed(seed)
    for _ in range(n_cases):
        n = random.randint(2, 4); m = random.randint(1, 5)
        A = np.random.randint(-4, 6, (m, n)).tolist()
        b = np.random.randint(-3, 12, m).tolist()
        c = np.random.randint(-3, 6, n).tolist()
        signs = [random.choice(['<=', '>=', '=']) for _ in range(m)]
        check(c, A, b, signs, random.random() < .5)


if __name__ == '__main__':
    test_examples(); test_random()
    print("Tutti i test superati.")
