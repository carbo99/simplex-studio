"""
SimplexStudio – Solver con Due Fasi e Visualizzazione 3D
=======================================================

Questo software:
1. Prende un insieme di vincoli di un problema di Programmazione Lineare (PL)
   con qualsiasi segno (≤, ≥, =) e termini noti positivi o negativi
2. Sistema automaticamente i segni e i termini noti negativi (b < 0)
3. Trova automaticamente una base ammissibile (Metodo a Due Fasi / Variabili Artificiali)
4. Crea il tableau del simplesso con la struttura corretta ad ogni iterazione
5. Visualizza la regione ammissibile in R³ (o R²) e il percorso del simplesso
6. Mostra ad ogni iterazione come cambia la soluzione di base ammissibile (BFS)

Copyright (c) 2026 Alessandro Carboni
carboni.alessandro@outlook.it
"""

import os
import tempfile
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl')

import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations


# ============================================================
#  Classe SimplexSolver
# ============================================================

class SimplexSolver:
    """
    Risolve un problema di PL generale con il metodo del simplesso (a Due Fasi se necessario).
    
    Supporta:
      • Massimizzazione / Minimizzazione
      • Vincoli ≤, ≥, =
      • Termini noti b positivi, nulli o negativi (normalizzati in automatico)
      • Variabili decisionali x ≥ 0
    """

    def __init__(self, c, A, b, signs=None, var_names=None, use_fractions=True, pivot_rule='dantzig', is_min=False):
        from exact_engine import ExactSimplex, number
        c = [number(v) for v in c]
        A = [[number(v) for v in row] for row in A]
        b = [number(v) for v in b]
        self.n, self.m = len(c), len(b)
        if not c or len(A) != self.m or any(len(row) != self.n for row in A):
            raise ValueError('Dimensioni incompatibili di c, A e b.')
        aliases = {'<=': '<=', '≤': '<=', '>=': '>=', '≥': '>=', '=': '='}
        signs = list(signs) if signs is not None else ['<='] * self.m
        if len(signs) != self.m or any(v not in aliases for v in signs):
            raise ValueError('Segni dei vincoli non validi.')
        self.signs_orig = [aliases[v] for v in signs]
        self.var_names = list(var_names) if var_names is not None else [f'x{i+1}' for i in range(self.n)]
        if len(self.var_names) != self.n or len(set(self.var_names)) != self.n:
            raise ValueError('I nomi delle variabili devono essere univoci e coerenti con c.')
        self.pivot_rule = (pivot_rule or 'dantzig').lower()
        if self.pivot_rule not in ('dantzig', 'bland'):
            raise ValueError('Regola pivot non valida.')
        self.is_min, self.use_fractions = is_min, use_fractions
        # Compatibility: c is the maximization cost, already negated by legacy
        # callers for min problems. Swift accepts the original objective instead.
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float).reshape(self.m, self.n)
        self.b_orig = np.array(b, dtype=float)
        if not all(np.isfinite(v).all() for v in (self.c_orig, self.A_orig, self.b_orig)):
            raise ValueError('Coefficienti fuori dall’intervallo visualizzabile.')
        self.history, self.redundant_rows, self.normalization_notes = [], [], []
        self.status = 'running'
        self.needs_phase1 = any(s != '<=' or bi < 0 for s, bi in zip(self.signs_orig, b))
        self._exact = ExactSimplex(c, A, b, self.signs_orig, self.var_names, self.pivot_rule, self._accept_exact)
        self.needs_phase1 = self._exact.phase == 1
        self.A_norm = np.array(self._exact.normal_A, dtype=float).reshape(self.m, self.n)
        self.b_norm = np.array(self._exact.normal_b, dtype=float)
        self.signs_norm = self._exact.normal_signs
        self.slack_cols = self._exact.slacks
        self.slack_names = [f's{i+1}' for i in range(len(self.slack_cols))]
        self.art_names = [name for name in self.all_var_names[self.n+len(self.slack_cols):]]
        for i, bi in enumerate(b):
            if bi < 0:
                self.normalization_notes.append(f'Vincolo {i+1}: RHS negativo, segno invertito.')

    def _accept_exact(self, engine, description, pivot, status):
        self.tableau = np.array(engine.t + [engine.obj], dtype=float)
        self.basis, self.all_var_names = list(engine.basis), list(engine.names)
        self.current_obj_label = '-w' if engine.phase == 1 else '-z'
        self.redundant_rows = list(engine.redundant)
        self.status = status
        self._save_state(description, pivot, engine.phase, status)
        state = self.history[-1]
        state['exact_tableau'] = [list(row) for row in engine.t + [engine.obj]]
        exact_solution = [Fraction(0)] * len(engine.names)
        for i, basic in enumerate(engine.basis):
            if basic < len(exact_solution):
                exact_solution[basic] = engine.t[i][0]
        state['exact_solution'] = exact_solution
        state['exact_z'] = engine.obj[0] if engine.phase == 1 else -engine.obj[0]
        state['dual_vector'] = np.array([v*f*(-1 if self.is_min else 1)
                                        for v, f in zip(engine.dual, engine.flips)], dtype=float)

    def _to_frac(self, val):
        if isinstance(val, Fraction):
            return str(val)
        if abs(val) < 1e-10:
            return "0"
        if abs(val - round(val)) < 1e-10:
            return str(int(round(val)))
        f = Fraction(val).limit_denominator(1000)
        return str(f)

    def _format_val(self, val):
        if self.use_fractions:
            return self._to_frac(val)
        else:
            val = float(val)
            if val == int(val):
                return str(int(val))
            return f"{val:.4f}"

    def get_basic_solution(self):
        """Restituisce il vettore di soluzione e il valore dell'obiettivo corrente."""
        total_vars = len(self.all_var_names)
        solution = np.zeros(total_vars)
        for i, var_idx in enumerate(self.basis):
            if var_idx < total_vars:
                solution[var_idx] = self.tableau[i, 0]
        if hasattr(self, 'current_obj_label') and self.current_obj_label == "-w":
            obj_val = self.tableau[self.m, 0]
        else:
            obj_val = -self.tableau[self.m, 0]
        return solution, obj_val

    def get_decision_variables(self):
        sol, z = self.get_basic_solution()
        return sol[:self.n], z

    def _save_state(self, description, pivot_info=None, phase=2, status=None):
        sol, z = self.get_basic_solution()
        dec_vars, _ = self.get_decision_variables()
        cur_status = status if status is not None else getattr(self, 'status', 'running')
        state = {
            'description': description,
            'phase': phase,
            'status': cur_status,
            'obj_label': getattr(self, 'current_obj_label', '-z'),
            'tableau': self.tableau.copy(),
            'basis': self.basis.copy(),
            'all_var_names': list(self.all_var_names),
            'solution': sol.copy(),
            'decision_vars': dec_vars.copy(),
            'z_value': z,
            'pivot_info': pivot_info
        }
        self.history.append(state)

    # ============================================================
    #  Stampa del Tableau (formato identico alla foto)
    # ============================================================

    def print_tableau(self, highlight_pivot=None, iteration=None, phase_name=None):
        m, n = self.m, self.n
        tableau = np.array(self.history[-1]['exact_tableau'], dtype=object)
        total_vars = len(self.all_var_names)
        obj_lbl = getattr(self, 'current_obj_label', '-z')

        if iteration is not None:
            tag = f" – {phase_name}" if phase_name else ""
            print(f"\n{'='*70}")
            print(f"  ITERAZIONE {iteration}{tag}")
            print(f"{'='*70}")

        col_widths = []
        for k in range(total_vars):
            w = max(6, len(self.all_var_names[k]) + 2)
            for i in range(m + 1):
                val_len = len(self._format_val(tableau[i, 1 + k]))
                w = max(w, val_len + 2)
            col_widths.append(w)

        z_width = max(6, len(obj_lbl) + 2)
        for i in range(m + 1):
            val_len = len(self._format_val(tableau[i, 0]))
            z_width = max(z_width, val_len + 2)

        margin = "    "

        # Intestazione colonne
        print()
        header = margin
        for k in range(total_vars):
            header += f" {self.all_var_names[k]:^{col_widths[k]}}"
        header += f"  │ {obj_lbl:^{z_width}}"
        print(header)

        # Riga obiettivo
        obj_str = margin
        for k in range(total_vars):
            val = self._format_val(tableau[m, 1 + k])
            if highlight_pivot and (m, 1 + k) == highlight_pivot:
                val = f"[{val}]"
            obj_str += f" {val:^{col_widths[k]}}"
        z_val = self._format_val(tableau[m, 0])
        if highlight_pivot and (m, 0) == highlight_pivot:
            z_val = f"[{z_val}]"
        obj_str += f"  │ {z_val:^{z_width}}"
        print(obj_str)

        # Separatore
        line_left_width = sum(col_widths) + total_vars
        separator = margin + "─" * line_left_width + "──┼─" + "─" * z_width
        print(separator)

        # Righe dei vincoli
        for i in range(m):
            row_str = margin
            for k in range(total_vars):
                val = self._format_val(tableau[i, 1 + k])
                if highlight_pivot and (i, 1 + k) == highlight_pivot:
                    val = f"[{val}]"
                row_str += f" {val:^{col_widths[k]}}"
            b_val = self._format_val(tableau[i, 0])
            if highlight_pivot and (i, 0) == highlight_pivot:
                b_val = f"[{b_val}]"
            row_str += f"  │ {b_val:^{z_width}}"
            print(row_str)

        dec_vars, z = self.history[-1]['exact_solution'][:self.n], self.history[-1]['exact_z']
        print()
        sol_parts = [f"{self.var_names[i]}={self._format_val(dec_vars[i])}" for i in range(self.n)]
        print(f"  Soluzione di Base Ammissibile: ({', '.join(sol_parts)})")
        print(f"  Valore {obj_lbl.replace('-', '')} = {self._format_val(z)}")
        basis_display = [self.all_var_names[b] for b in self.basis if b < len(self.all_var_names)]
        print(f"  Variabili in base: {basis_display}")
        print()

    # ============================================================
    #  Algoritmo di Pivot
    # ============================================================

    def solve(self, verbose=True, plot3d=True, save_figures=False, output_dir=None, max_iterations=500):
        if not isinstance(max_iterations, int) or max_iterations < 0:
            raise ValueError('Il limite iterazioni deve essere un intero non negativo.')
        if self.status == 'running':
            self.status = self._exact.solve(max_iterations)
        if verbose:
            for i, state in enumerate(self.history):
                print(f"{i}: {state['description']}")
            self.print_tableau()
        if self.status != 'optimal':
            return None
        if save_figures:
            os.makedirs(output_dir or '.', exist_ok=True)
        if plot3d and self.n == 3:
            self._plot_3d_feasible_region(save_figures, output_dir or '.')
        elif plot3d and self.n == 2:
            self._plot_2d_feasible_region(save_figures, output_dir or '.')
        return self.get_decision_variables()

    def get_dual_solution(self):
        """
        Calcola i valori ottimi delle variabili duali (prezzi ombra) y*
        associati a ciascun vincolo originale.
        Disponibile solo se il problema ha raggiunto una soluzione ottima finita.
        """
        if getattr(self, 'status', 'optimal') != 'optimal':
            return None
        return self.get_dual_vector_at_step(len(self.history) - 1) if self.history else None

    def get_dual_vector_at_step(self, step_idx):
        if not 0 <= step_idx < len(self.history):
            return None
        state = self.history[step_idx]
        # Phase 1 multipliers belong to the auxiliary problem, not the primal.
        if state['phase'] == 1:
            return None
        return state['dual_vector'].copy()

    def get_dual_formulation(self, is_primal_min=None):
        """
        Restituisce la formulazione esatta del problema duale associato.
        Ritorna: obj_type, b_coeff, A_dual, c_bounds, y_signs
        """
        if is_primal_min is None:
            is_primal_min = self.is_min
        m, n = self.m, self.n
        A = self.A_orig
        b = self.b_orig
        c = -self.c_orig if is_primal_min else self.c_orig

        dual_obj_type = "max" if is_primal_min else "min"

        # Matrice trasposta per i vincoli duali: n vincoli, m variabili y
        A_dual = A.T
        b_dual = c.copy()

        # Segni dei vincoli duali e segni delle variabili duali y
        if not is_primal_min:
            # Primale: max c^T x, s.t. A_i x (<=, >=, =) b_i, x >= 0
            # Duale: min b^T y, s.t. A^T y >= c, y_i (>= 0 se <=, <= 0 se >=, libera se =)
            dual_constr_signs = ['>='] * n
            y_signs = []
            for s in self.signs_orig:
                if s == '<=':
                    y_signs.append('>= 0')
                elif s == '>=':
                    y_signs.append('<= 0')
                else:
                    y_signs.append('libera')
        else:
            # Primale: min c^T x, s.t. A_i x (<=, >=, =) b_i, x >= 0
            # Duale: max b^T y, s.t. A^T y <= c, y_i (<= 0 se <=, >= 0 se >=, libera se =)
            dual_constr_signs = ['<='] * n
            y_signs = []
            for s in self.signs_orig:
                if s == '>=':
                    y_signs.append('>= 0')
                elif s == '<=':
                    y_signs.append('<= 0')
                else:
                    y_signs.append('libera')

        return {
            'obj_type': dual_obj_type,
            'obj_coeffs': b,
            'A': A_dual,
            'b': b_dual,
            'constr_signs': dual_constr_signs,
            'y_signs': y_signs
        }

    # ============================================================
    #  Geometria e Vertici (supporta ≤, ≥, =)
    # ============================================================

    def _is_point_feasible(self, x):
        """Verifica se un punto x rispetta tutti i vincoli originali e x >= 0."""
        if len(x) != self.n or not np.isfinite(x).all():
            return False
        if any(xj < -1e-7 for xj in x):
            return False
        for i in range(self.m):
            val = np.dot(self.A_orig[i], x)
            s = self.signs_orig[i]
            bi = self.b_orig[i]
            scale = max(float(np.max(np.abs(self.A_orig[i]))), abs(bi)) or 1.0
            tolerance = 1e-7 * scale
            if s == '<=' and val > bi + tolerance:
                return False
            elif s == '>=' and val < bi - tolerance:
                return False
            elif s == '=' and abs(val - bi) > tolerance:
                return False
        return True

    def find_all_vertices(self):
        """Trova tutti i vertici ammissibili della regione del poliedro."""
        from geometry import vertices
        return vertices(self._exact.A, self._exact.b, self.signs_orig, self.n)

    def display_vertices(self, bounds):
        from geometry import vertices
        return vertices(self._exact.A, self._exact.b, self.signs_orig, self.n, bounds=bounds)

    # ============================================================
    #  Visualizzazione 3D
    # ============================================================

    def _plot_3d_feasible_region(self, save_figures=False, output_dir="."):
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        self._draw_feasible_region_3d(ax1)
        self._draw_simplex_path(ax1)
        ax1.set_xlabel(self.var_names[0], fontsize=12)
        ax1.set_ylabel(self.var_names[1], fontsize=12)
        ax1.set_zlabel(self.var_names[2], fontsize=12)
        ax1.set_title("Regione ammissibile e percorso del simplesso", fontsize=13, pad=15)

        ax2 = fig.add_subplot(122, projection='3d')
        self._draw_feasible_region_3d(ax2, alpha=0.05)
        self._draw_iterations_3d(ax2)
        ax2.set_xlabel(self.var_names[0], fontsize=12)
        ax2.set_ylabel(self.var_names[1], fontsize=12)
        ax2.set_zlabel(self.var_names[2], fontsize=12)
        ax2.set_title("Vertici visitati ad ogni iterazione", fontsize=13, pad=15)

        plt.tight_layout()
        if save_figures:
            path = os.path.join(output_dir, "simplex_3d.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\n  Figura salvata: {path}")
        plt.show()

    def _draw_feasible_region_3d(self, ax, alpha=0.15):
        from geometry import draw_region
        verts = self.find_all_vertices()
        extent = max(float(verts.max()) * 1.3, 5) if len(verts) else 10
        draw_region(ax, self.display_vertices([(0, extent)] * 3), alpha)
        ax.set_xlim(0, extent); ax.set_ylim(0, extent); ax.set_zlim(0, extent)
        ax.set_title('Regione nella finestra di visualizzazione')

    def _draw_simplex_path(self, ax):
        points = [s['decision_vars'][:3] for s in self.history if len(s['decision_vars']) >= 3]
        if not points:
            return
        pts = np.array(points)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                'r-o', linewidth=2.5, markersize=8, zorder=10,
                markerfacecolor='red', markeredgecolor='darkred')
        ax.scatter(*pts[0], c='green', s=150, marker='^', zorder=11,
                   edgecolors='darkgreen', linewidths=2, label='Start')
        ax.scatter(*pts[-1], c='gold', s=200, marker='*', zorder=11,
                   edgecolors='darkorange', linewidths=2, label='Ottimo' if self.status == 'optimal' else 'Ultimo passo')
        for i, pt in enumerate(pts):
            ax.text(pt[0] + 0.1, pt[1] + 0.1, pt[2] + 0.1,
                    f"It.{i}", fontsize=9, fontweight='bold', color='red')
        ax.legend(fontsize=10)

    def _draw_iterations_3d(self, ax):
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.history)))
        for i, state in enumerate(self.history):
            dec = state['decision_vars']
            if len(dec) >= 3:
                pt = dec[:3]
                ax.scatter(*pt, c=[colors[i]], s=120, zorder=10,
                           edgecolors='black', linewidths=1.5)
                label = f"It.{i}: ({self._format_val(pt[0])}, {self._format_val(pt[1])}, {self._format_val(pt[2])})"
                ax.text(pt[0] + 0.15, pt[1] + 0.15, pt[2] + 0.15,
                        label, fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i],
                                  alpha=0.7, edgecolor='black'))

    # ============================================================
    #  Visualizzazione 2D
    # ============================================================

    def _plot_2d_feasible_region(self, save_figures=False, output_dir="."):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        verts = self.find_all_vertices()
        max_val = max(verts.max() * 1.3, 5) if len(verts) > 0 else 10

        self._draw_feasible_region_2d(ax1, max_val, max_val)
        self._draw_simplex_path_2d(ax1)
        ax1.set_xlabel(self.var_names[0], fontsize=12)
        ax1.set_ylabel(self.var_names[1], fontsize=12)
        ax1.set_title("Regione ammissibile e percorso del simplesso", fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        self._draw_feasible_region_2d(ax2, max_val, max_val, alpha=0.1)
        self._draw_iterations_2d(ax2)
        ax2.set_xlabel(self.var_names[0], fontsize=12)
        ax2.set_ylabel(self.var_names[1], fontsize=12)
        ax2.set_title("Vertici visitati ad ogni iterazione", fontsize=13)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_figures:
            path = os.path.join(output_dir, "simplex_2d.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\n  Figura salvata: {path}")
        plt.show()

    def _draw_feasible_region_2d(self, ax, max_x, max_y, alpha=0.3):
        x_range = np.linspace(0, max_x, 400)
        for i in range(self.m):
            a1, a2 = self.A_orig[i, 0], self.A_orig[i, 1]
            bi = self.b_orig[i]
            s = '≤' if self.signs_orig[i] == '<=' else ('≥' if self.signs_orig[i] == '>=' else '=')
            if a2 != 0:
                y_line = (bi - a1 * x_range) / a2
                lbl = f"{self._format_val(a1)}{self.var_names[0]}+{self._format_val(a2)}{self.var_names[1]} {s} {self._format_val(bi)}"
                ax.plot(x_range, y_line, label=lbl, linewidth=1.5)
            elif a1 != 0:
                if a1 < 0:
                    s = {'≤': '≥', '≥': '≤', '=': '='}[s]
                ax.axvline(x=bi/a1, label=f"{self.var_names[0]} {s} {self._format_val(bi/a1)}", linewidth=1.5)

        from geometry import draw_region
        draw_region(ax, self.display_vertices([(0, max_x), (0, max_y)]), alpha)

        ax.set_xlim([-0.5, max_x])
        ax.set_ylim([-0.5, max_y])

    def _draw_simplex_path_2d(self, ax):
        points = [s['decision_vars'][:2] for s in self.history]
        pts = np.array(points)
        ax.plot(pts[:, 0], pts[:, 1], 'r-o', linewidth=2.5, markersize=8,
                markerfacecolor='red', markeredgecolor='darkred', zorder=10, label='Percorso')
        ax.scatter(*pts[0], c='green', s=150, marker='^', zorder=11,
                   edgecolors='darkgreen', linewidths=2, label='Start')
        ax.scatter(*pts[-1], c='gold', s=200, marker='*', zorder=11,
                   edgecolors='darkorange', linewidths=2, label='Ottimo' if self.status == 'optimal' else 'Ultimo passo')
        for i, pt in enumerate(pts):
            ax.annotate(f"It.{i}", pt, textcoords="offset points",
                        xytext=(10, 10), fontsize=9, fontweight='bold', color='red')

    def _draw_iterations_2d(self, ax):
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.history)))
        for i, state in enumerate(self.history):
            pt = state['decision_vars'][:2]
            ax.scatter(*pt, c=[colors[i]], s=120, zorder=10,
                       edgecolors='black', linewidths=1.5)
            label = f"It.{i}: ({self._format_val(pt[0])}, {self._format_val(pt[1])})"
            ax.annotate(label, pt, textcoords="offset points",
                        xytext=(10, 10), fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i],
                                  alpha=0.7, edgecolor='black'))

    def plot_iteration_by_iteration(self, save_figures=False, output_dir="."):
        if self.n not in (2, 3):
            return
        for idx, state in enumerate(self.history):
            fig = plt.figure(figsize=(10, 8))
            if self.n == 3:
                ax = fig.add_subplot(111, projection='3d')
                self._draw_feasible_region_3d(ax, alpha=0.12)
                for j in range(idx + 1):
                    pt = self.history[j]['decision_vars'][:3]
                    color = 'green' if j == 0 else ('gold' if j == idx and idx == len(self.history) - 1 else 'red')
                    marker = '^' if j == 0 else ('*' if j == idx and idx == len(self.history) - 1 else 'o')
                    size = 150 if j in (0, idx) else 80
                    ax.scatter(*pt, c=color, s=size, marker=marker, zorder=10,
                               edgecolors='black', linewidths=1.5)
                    ax.text(pt[0] + 0.1, pt[1] + 0.1, pt[2] + 0.1,
                            f"It.{j}", fontsize=10, fontweight='bold')
                    if j > 0:
                        prev = self.history[j - 1]['decision_vars'][:3]
                        ax.plot([prev[0], pt[0]], [prev[1], pt[1]], [prev[2], pt[2]], 'r-', linewidth=2)
                ax.set_xlabel(self.var_names[0], fontsize=12)
                ax.set_ylabel(self.var_names[1], fontsize=12)
                ax.set_zlabel(self.var_names[2], fontsize=12)
            else:
                ax = fig.add_subplot(111)
                verts = self.find_all_vertices()
                max_v = max(verts.max() * 1.3, 5) if len(verts) > 0 else 10
                self._draw_feasible_region_2d(ax, max_v, max_v, alpha=0.15)
                for j in range(idx + 1):
                    pt = self.history[j]['decision_vars'][:2]
                    color = 'green' if j == 0 else ('gold' if j == idx and idx == len(self.history) - 1 else 'red')
                    marker = '^' if j == 0 else ('*' if j == idx and idx == len(self.history) - 1 else 'o')
                    size = 150 if j in (0, idx) else 80
                    ax.scatter(*pt, c=color, s=size, marker=marker, zorder=10,
                               edgecolors='black', linewidths=1.5)
                    ax.annotate(f"It.{j}", pt, textcoords="offset points",
                                xytext=(10, 10), fontsize=10, fontweight='bold')
                    if j > 0:
                        prev = self.history[j - 1]['decision_vars'][:2]
                        ax.plot([prev[0], pt[0]], [prev[1], pt[1]], 'r-', linewidth=2)
                ax.set_xlabel(self.var_names[0], fontsize=12)
                ax.set_ylabel(self.var_names[1], fontsize=12)
                ax.grid(True, alpha=0.3)

            dec = state['decision_vars']
            title = state['description']
            sol_str = ", ".join(f"{self.var_names[i]}={self._format_val(dec[i])}" for i in range(self.n))
            z_str = self._format_val(state['z_value'])
            lbl = state.get('obj_label', '-z').replace('-', '')
            ax.set_title(f"{title}\nSoluzione di Base Ammissibile: ({sol_str}), {lbl} = {z_str}", fontsize=12, pad=15)
            plt.tight_layout()
            if save_figures:
                path = os.path.join(output_dir, f"simplex_iter_{idx}.png")
                plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.show()


# ============================================================
#  Esempi
# ============================================================

def esempio_3_variabili():
    """
    Esempio in R3:
      max z = 3x1 + 2x2 + 4x3
      s.t.
          x1 +  x2 +  x3 <= 4
         2x1 +  x2 + 3x3 <= 6
          x1 + 2x2 +  x3 <= 5
          x1, x2, x3 >= 0
    """
    c = [3, 2, 4]
    A = [
        [1, 1, 1],
        [2, 1, 3],
        [1, 2, 1]
    ]
    b = [4, 6, 5]
    signs = ['<=', '<=', '<=']
    solver = SimplexSolver(c, A, b, signs=signs)
    solver.solve(verbose=True, plot3d=True)
    return solver

def esempio_2_variabili():
    """
    Esempio in R2:
      max z = x1 + 3x2
      s.t.
         6x1 + 10x2 <= 30
         3x1 +  2x2 >= 6
          x1 -  2x2 >= -1
                x2 >= 1/2
          x1, x2 >= 0
    """
    c = [1, 3]
    A = [
        [6, 10],
        [3, 2],
        [1, -2],
        [0, 1]
    ]
    b = [30, 6, -1, 0.5]
    signs = ['<=', '>=', '>=', '>=']
    solver = SimplexSolver(c, A, b, signs=signs)
    solver.solve(verbose=True, plot3d=True)
    return solver

def _input_int(prompt, min_val=1, max_val=100):
    while True:
        try:
            raw = input(prompt).strip()
            if not raw:
                continue
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Il valore deve essere tra {min_val} e {max_val}.")
        except ValueError:
            print("  Inserisci un numero intero valido.")

def _input_float(prompt):
    while True:
        try:
            raw = input(prompt).strip()
            if not raw:
                continue
            if '/' in raw:
                p = raw.split('/')
                return float(Fraction(int(p[0]), int(p[1])))
            return float(raw)
        except (ValueError, ZeroDivisionError):
            print("  Inserisci un numero valido (es. 3, 1.5, 1/2).")

def esempio_personalizzato():
    print("\n" + "─" * 70)
    print("  INSERISCI IL TUO PROBLEMA DI PL (Supporta <=, >=, = e b qualsiasi)")
    print("─" * 70)
    n = _input_int("  Numero di variabili decisionali: ", 1, 20)
    m = _input_int("  Numero di vincoli: ", 1, 50)

    print(f"\n  Inserisci i coefficienti della funzione obiettivo (max z):")
    c = [_input_float(f"    c{i+1} = ") for i in range(n)]

    A, b, signs = [], [], []
    for i in range(m):
        print(f"\n  Vincolo {i+1}:")
        row = [_input_float(f"    a{i+1},{j+1} = ") for j in range(n)]
        s_in = input("    Segno (<=, >=, = oppure premere Invio per <=): ").strip()
        if s_in in ('>=', '≥'):
            s = '>='
        elif s_in == '=':
            s = '='
        else:
            s = '<='
        bi = _input_float(f"    Termine noto b{i+1} = ")
        A.append(row); b.append(bi); signs.append(s)

    solver = SimplexSolver(c, A, b, signs=signs)
    solver.solve(verbose=True, plot3d=True)
    return solver

if __name__ == "__main__":
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│         SimplexStudio – Programmazione Lineare Interattiva       │")
    print("│   Due Fasi Automatiche • Segni <=, >=, = • Tableau & Grafica     │")
    print("└──────────────────────────────────────────────────────────────────┘")
    print("\n  [1] Esempio in R³\n  [2] Esempio in R²\n  [3] Inserisci il tuo problema\n")
    while True:
        choice = input("  Scelta (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            break
    if choice == "1":
        esempio_3_variabili()
    elif choice == "2":
        esempio_2_variabili()
    else:
        esempio_personalizzato()
