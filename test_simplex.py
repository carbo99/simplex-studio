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


def _check_basis_consistent(s):
    """Nessuna variabile reale compare due volte in base, in nessun passo."""
    for h in s.history:
        real = [b for b in h['basis'] if b < len(h['all_var_names'])]
        assert len(real) == len(set(real)), ("base con indici duplicati", h['basis'])


def test_redundant_regression():
    # Controesempio: V2 e V3 coincidono -> riga ridondante dopo la Fase 1.
    c, A, b, signs = [1, -2], [[2, 1], [-3, 3], [-1, 1]], [7, 6, 2], ['<=', '=', '=']
    s = SimplexSolver(c, A, b, signs=signs)
    r = s.solve(verbose=False, plot3d=False)
    assert s.status == 'optimal' and abs(r[1] + 4) < 1e-9
    y = s.get_dual_solution()
    assert abs(float(np.dot(b, y)) + 4) < 1e-9, ("dualita forte", y)
    assert len(s.redundant_rows) == 1
    _check_basis_consistent(s)


def test_redundant_random(n_cases=1500, seed=3):
    """Problemi ammissibili con righe linearmente dipendenti costruite apposta."""
    random.seed(seed); np.random.seed(seed)
    for _ in range(n_cases):
        n = random.randint(2, 5); m0 = random.randint(1, 3)
        A = np.random.randint(-3, 5, (m0, n)).tolist(); x0 = np.random.randint(0, 4, n)
        b = [float(np.dot(r, x0)) for r in A]
        signs = [random.choice(['=', '=', '<=']) for _ in range(m0)]
        for _k in range(random.randint(1, 2)):
            i, j = random.sample(range(m0), 2) if m0 > 1 else (0, 0)
            k = random.choice([1, 2, -1, 3])
            row = [k * A[i][t] + (A[j][t] if i != j else 0) for t in range(n)]
            A.append(row); b.append(float(np.dot(row, x0))); signs.append('=')
        c = np.random.randint(-3, 6, n).tolist(); is_min = random.random() < .5
        cc = [-v for v in c] if is_min else c
        s = SimplexSolver(cc, A, b, signs=signs, is_min=is_min)
        r = s.solve(verbose=False, plot3d=False)
        ref = _scipy(c, A, b, signs, is_min)
        ref_status = {0: 'optimal', 2: 'infeasible', 3: 'unbounded'}.get(ref.status, 'other')
        assert s.status == ref_status
        _check_basis_consistent(s)
        if s.status == 'optimal':
            z = -r[1] if is_min else r[1]
            assert abs(z - (ref.fun if is_min else -ref.fun)) < 1e-6
            y = s.get_dual_solution()
            assert abs(float(np.dot(b, y)) - z) < 1e-6, ("dualita forte", y, z)


def test_bland():
    """Stessa verifica contro SciPy usando la regola di Bland (prima non coperta)."""
    random.seed(2); np.random.seed(2)
    for _ in range(600):
        n = random.randint(2, 5); m = random.randint(1, 6)
        A = np.random.randint(-4, 6, (m, n)).tolist(); b = np.random.randint(-3, 12, m).tolist()
        c = np.random.randint(-3, 6, n).tolist()
        signs = [random.choice(['<=', '>=', '=']) for _ in range(m)]
        s = SimplexSolver(c, A, b, signs=signs, pivot_rule='bland')
        r = s.solve(verbose=False, plot3d=False)
        ref = _scipy(c, A, b, signs, False)
        assert s.status == {0: 'optimal', 2: 'infeasible', 3: 'unbounded'}.get(ref.status, 'other')
        if s.status == 'optimal':
            assert abs(r[1] + ref.fun) < 1e-6


if __name__ == '__main__':
    test_examples(); test_random(); test_redundant_regression(); test_redundant_random(); test_bland()
    print("Tutti i test superati.")
