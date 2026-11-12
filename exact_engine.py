"""Exact two-phase simplex kernel. No numerical tolerances in pivot decisions."""
from fractions import Fraction
import math


def number(value):
    try:
        if isinstance(value, str):
            value = value.strip().replace(',', '.')
            parts = value.split('/')
            if len(parts) == 2:
                return number(parts[0]) / number(parts[1])
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('I coefficienti devono essere finiti.')
        return Fraction(str(value))
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f'Numero non valido: {value!r}') from exc


class ExactSimplex:
    def __init__(self, c, A, b, signs, names, rule, emit):
        self.c, self.A, self.b = c, A, b
        self.signs, self.rule, self.emit = signs, rule, emit
        self.n, self.original_m = len(c), len(b)
        self.names = list(names)
        self.slacks, arts, self.basis = [], [], []
        self.flips = [-1 if v < 0 else 1 for v in b]
        self.normal_A = [[v * f for v in row] for row, f in zip(A, self.flips)]
        self.normal_b = [v * f for v, f in zip(b, self.flips)]
        self.normal_signs = [({'<=': '>=', '>=': '<=', '=': '='}[s] if f < 0 else s)
                             for s, f in zip(signs, self.flips)]
        for i, s in enumerate(self.normal_signs):
            if s != '=':
                self.slacks.append((i, 1 if s == '<=' else -1))
        self.names += [f's{i+1}' for i in range(len(self.slacks))]
        self.real_count = len(self.names)
        for i, s in enumerate(self.normal_signs):
            if s != '<=':
                arts.append(i)
        self.names += [f'a{i+1}' for i in range(len(arts))]
        self.t = [[Fraction(v) for v in [rhs] + row + [0] * (len(self.names)-self.n)]
                  for row, rhs in zip(self.normal_A, self.normal_b)]
        self.transform = [[Fraction(int(i == j)) for j in range(len(b))] for i in range(len(b))]
        for k, (r, sign) in enumerate(self.slacks):
            self.t[r][1+self.n+k] = Fraction(sign)
        for k, r in enumerate(arts):
            self.t[r][1+self.real_count+k] = Fraction(1)
        for i, s in enumerate(self.normal_signs):
            self.basis.append(self.real_count + arts.index(i) if s != '<=' else
                              self.n + next(k for k, (r, _) in enumerate(self.slacks) if r == i))
        self.phase = 1 if arts else 2
        self.redundant = []
        self.iterations = 0
        self.obj = []
        self.dual = []
        self.objective([Fraction(0)]*self.real_count + [Fraction(-1)]*len(arts) if arts else self.cost())
        self.record('Tableau iniziale')

    def cost(self):
        return self.c + [Fraction(0)] * (self.real_count-self.n)

    def objective(self, cost):
        self.obj = [Fraction(0)] + list(cost)
        self.dual = [Fraction(0)] * self.original_m
        for i, basic in enumerate(self.basis):
            cb = cost[basic]
            self.obj = [a-cb*v for a, v in zip(self.obj, self.t[i])]
            self.dual = [a+cb*v for a, v in zip(self.dual, self.transform[i])]

    def record(self, text, pivot=None, status='running'):
        self.emit(self, text, pivot, status)

    def pivot(self, row, col):
        p = self.t[row][col]
        self.t[row] = [v/p for v in self.t[row]]
        self.transform[row] = [v/p for v in self.transform[row]]
        for i in range(len(self.t)):
            if i != row:
                f = self.t[i][col]
                self.t[i] = [a-f*v for a, v in zip(self.t[i], self.t[row])]
                self.transform[i] = [a-f*v for a, v in zip(self.transform[i], self.transform[row])]
        f = self.obj[col]
        self.obj = [a-f*v for a, v in zip(self.obj, self.t[row])]
        self.dual = [a+f*v for a, v in zip(self.dual, self.transform[row])]
        self.basis[row] = col-1

    def optimize(self, limit):
        seen = set()
        while True:
            key = tuple(self.basis)
            if key in seen:
                self.rule = 'bland'
            seen.add(key)
            candidates = [j for j in range(1, len(self.obj)) if self.obj[j] > 0]
            if not candidates:
                return 'optimal'
            if self.iterations >= limit:
                return 'max_iterations'
            col = candidates[0] if self.rule == 'bland' else max(candidates, key=lambda j: self.obj[j])
            rows = [i for i, row in enumerate(self.t) if row[col] > 0]
            if not rows:
                return 'unbounded'
            row = min(rows, key=lambda i: (self.t[i][0]/self.t[i][col], self.basis[i]))
            info = dict(entering=self.names[col-1], leaving=self.names[self.basis[row]],
                        pivot_row=row, pivot_col=col, pivot_value=float(self.t[row][col]))
            self.pivot(row, col)
            self.iterations += 1
            self.record(f'Fase {self.phase} – Iterazione {self.iterations}: {info["entering"]} entra, {info["leaving"]} esce', info)

    def solve(self, limit=500):
        if self.phase == 1:
            status = self.optimize(limit)
            if status != 'optimal':
                self.record('Limite iterazioni Fase 1', status=status)
                return status
            if self.obj[0] != 0:
                self.record('Problema Inammissibile', status='infeasible')
                return 'infeasible'
            for i in range(len(self.t)):
                if self.basis[i] >= self.real_count:
                    cols = [j for j in range(1, self.real_count+1)
                            if j-1 not in self.basis and self.t[i][j] != 0]
                    if cols:
                        self.pivot(i, cols[0])
                        self.record('Uscita artificiale dalla base')
                    else:
                        self.redundant.append(i)
            # Retain zero rows to preserve the public tableau dimensions. Duals use
            # row transformations, never row positions as original constraint IDs.
            for i in self.redundant:
                self.basis[i] = self.real_count + i
            self.t = [row[:self.real_count+1] for row in self.t]
            self.names = self.names[:self.real_count]
            self.phase = 2
            cost = self.cost()
            self.obj = [Fraction(0)] + cost
            self.dual = [Fraction(0)] * self.original_m
            for i, basic in enumerate(self.basis):
                cb = cost[basic] if basic < len(cost) else 0
                self.obj = [a-cb*v for a, v in zip(self.obj, self.t[i])]
                self.dual = [a+cb*v for a, v in zip(self.dual, self.transform[i])]
            self.record('Inizio Fase 2 (base ammissibile)')
        status = self.optimize(limit)
        self.record({'optimal': 'Soluzione ottima', 'unbounded': 'Problema illimitato',
                     'max_iterations': 'Limite iterazioni raggiunto'}[status], status=status)
        return status
