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
        """
        Parametri
        ----------
        c : lista/array        – coefficienti della funzione obiettivo (da massimizzare)
        A : matrice             – coefficienti dei vincoli (m x n)
        b : lista/array         – termini noti dei vincoli (m,)
        signs : lista di str    – segno di ogni vincolo ('<=', '>=', '=' o '≤', '≥')
        var_names : lista       – nomi delle variabili decisionali (default: x1, x2, ...)
        use_fractions : bool    – se True usa frazioni esatte per il display
        pivot_rule : str        – regola di pivot: 'dantzig' (default classico) o 'bland' (anti-ciclo)
        is_min : bool           – se True il problema originale era di minimizzazione
        """
        self.is_min = is_min
        self.c_orig = np.array(c, dtype=float)
        self.A_orig = np.array(A, dtype=float)
        self.b_orig = np.array(b, dtype=float)
        self.m, self.n = self.A_orig.shape
        self.use_fractions = use_fractions
        self.pivot_rule = pivot_rule.lower() if pivot_rule else 'dantzig'
        self.status = 'running'

        # Salva segni originali per la visualizzazione geometrica
        if signs is None:
            self.signs_orig = ['<='] * self.m
        else:
            self.signs_orig = [
                '<=' if s in ('<=', '≤') else ('>=' if s in ('>=', '≥') else '=')
                for s in signs
            ]

        # Nomi delle variabili decisionali
        if var_names is None:
            self.var_names = [f"x{i+1}" for i in range(self.n)]
        else:
            self.var_names = list(var_names)

        # ── Normalizzazione automatica dei vincoli ──
        # Se un vincolo ha b_i < 0, moltiplica per -1 e inverte il segno
        self.A_norm = []
        self.b_norm = []
        self.signs_norm = []
        self.normalization_notes = []

        for i in range(self.m):
            row = list(self.A_orig[i])
            bi = float(self.b_orig[i])
            s = self.signs_orig[i]

            if bi < -1e-10:
                row = [-val for val in row]
                bi = -bi
                s_new = '<=' if s == '>=' else ('>=' if s == '<=' else '=')
                self.normalization_notes.append(
                    f"Vincolo {i+1}: b={self._format_val(self.b_orig[i])} < 0 → moltiplicato per -1 (segno invertito in {s_new})"
                )
                s = s_new
            elif abs(bi) < 1e-10:
                bi = 0.0

            self.A_norm.append(row)
            self.b_norm.append(bi)
            self.signs_norm.append(s)

        self.A_norm = np.array(self.A_norm, dtype=float)
        self.b_norm = np.array(self.b_norm, dtype=float)

        # ── Configurazione delle variabili (slack, surplus, artificiali) ──
        self._setup_variables()

        # ── Costruzione tableau iniziale ──
        self._build_initial_tableau()

        # Storico delle iterazioni
        self.history = []
        desc = "Tableau iniziale (Fase 1)" if self.needs_phase1 else "Tableau iniziale"
        self._save_state(desc, phase=(1 if self.needs_phase1 else 2))

    def _setup_variables(self):
        """Identifica variabili di slack, surplus e artificiali necessarie."""
        self.slack_names = []
        self.art_names = []
        self.slack_cols = []  # (row_index, coeff)
        self.art_cols = []    # (row_index, coeff)
        self.art_rows = []

        s_count = 1
        a_count = 1

        for i, s in enumerate(self.signs_norm):
            if s == '<=':
                name = f"s{s_count}"
                s_count += 1
                self.slack_names.append(name)
                self.slack_cols.append((i, 1.0))
            elif s == '>=':
                name_s = f"s{s_count}"
                s_count += 1
                self.slack_names.append(name_s)
                self.slack_cols.append((i, -1.0))  # surplus con segno -1

                name_a = f"a{a_count}"
                a_count += 1
                self.art_names.append(name_a)
                self.art_cols.append((i, 1.0))    # artificiale con segno +1
                self.art_rows.append(i)
            elif s == '=':
                name_a = f"a{a_count}"
                a_count += 1
                self.art_names.append(name_a)
                self.art_cols.append((i, 1.0))
                self.art_rows.append(i)

        self.needs_phase1 = len(self.art_names) > 0
        self.all_var_names = self.var_names + self.slack_names + self.art_names

    def _build_initial_tableau(self):
        """Costruisce il tableau iniziale (Fase 1 se ci sono artificiali, altrimenti Fase 2)."""
        m, n = self.m, self.n
        total_vars = len(self.all_var_names)

        self.tableau = np.zeros((m + 1, total_vars + 1))

        # Riempi termini noti b e coefficienti A
        for i in range(m):
            self.tableau[i, 0] = self.b_norm[i]
            self.tableau[i, 1:1+n] = self.A_norm[i]

        # Colonne slack/surplus
        col = 1 + n
        for (r, coeff) in self.slack_cols:
            self.tableau[r, col] = coeff
            col += 1

        # Colonne artificiali
        for (r, coeff) in self.art_cols:
            self.tableau[r, col] = coeff
            col += 1

        # Base iniziale
        self.basis = []
        s_idx = 0
        a_idx = 0
        for i, s in enumerate(self.signs_norm):
            if s == '<=':
                self.basis.append(n + s_idx)
                s_idx += 1
            elif s == '>=':
                s_idx += 1
                self.basis.append(n + len(self.slack_names) + a_idx)
                a_idx += 1
            elif s == '=':
                self.basis.append(n + len(self.slack_names) + a_idx)
                a_idx += 1

        # Riga obiettivo
        if self.needs_phase1:
            # Fase 1: minimizza w = sum(a_i) <=> massimizza -w = -sum(a_i)
            # In forma canonica rispetto alle variabili artificiali in base:
            for r in self.art_rows:
                self.tableau[m, 0] += self.tableau[r, 0]
                self.tableau[m, 1:] += self.tableau[r, 1:]
            art_start = 1 + n + len(self.slack_names)
            self.tableau[m, art_start:] = 0.0
            self.current_obj_label = "-w"
        else:
            # Fase 2 diretta
            self.tableau[m, 0] = 0.0
            self.tableau[m, 1:1+n] = self.c_orig
            self.current_obj_label = "-z"

    def _to_frac(self, val):
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
                val_len = len(self._format_val(self.tableau[i, 1 + k]))
                w = max(w, val_len + 2)
            col_widths.append(w)

        z_width = max(6, len(obj_lbl) + 2)
        for i in range(m + 1):
            val_len = len(self._format_val(self.tableau[i, 0]))
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
            val = self._format_val(self.tableau[m, 1 + k])
            if highlight_pivot and (m, 1 + k) == highlight_pivot:
                val = f"[{val}]"
            obj_str += f" {val:^{col_widths[k]}}"
        z_val = self._format_val(self.tableau[m, 0])
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
                val = self._format_val(self.tableau[i, 1 + k])
                if highlight_pivot and (i, 1 + k) == highlight_pivot:
                    val = f"[{val}]"
                row_str += f" {val:^{col_widths[k]}}"
            b_val = self._format_val(self.tableau[i, 0])
            if highlight_pivot and (i, 0) == highlight_pivot:
                b_val = f"[{b_val}]"
            row_str += f"  │ {b_val:^{z_width}}"
            print(row_str)

        dec_vars, z = self.get_decision_variables()
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

    def _find_pivot_column(self, max_col=None):
        """
        Trova la colonna pivot (variabile entrante).
        Se self.pivot_rule == 'bland':
            Regola di Bland: seleziona la prima colonna con costo ridotto positivo
            (indice minimo della variabile candidata), prevenendo cicli e stalli.
        Se self.pivot_rule == 'dantzig':
            Regola classica di Dantzig: seleziona la colonna con costo ridotto positivo massimo.
        """
        if max_col is None:
            max_col = self.tableau.shape[1]
        
        candidates = []
        for j in range(1, max_col):
            val = self.tableau[self.m, j]
            if val > 1e-10:
                candidates.append((j, val))

        if not candidates:
            return -1

        if self.pivot_rule == 'dantzig':
            return max(candidates, key=lambda item: item[1])[0]
        else:
            # Regola di Bland (minimo indice di variabile / prima colonna j ammissibile)
            return candidates[0][0]

    def _find_pivot_row(self, pivot_col):
        """
        Trova la riga pivot (variabile uscente) con la regola del rapporto minimo b_i / a_ij > 0.
        In caso di parità (tie) dovuta a degenerazione (più righe con lo stesso rapporto minimo):
        - Applica la regola di Bland: sceglie la riga la cui variabile di base associata
          ha indice minimo (self.basis[i]), garantendo matematicamente la convergenza ed escludendo ciclaggi.
        """
        min_ratio = float('inf')
        candidates = []

        for i in range(self.m):
            a_ij = self.tableau[i, pivot_col]
            if a_ij > 1e-10:
                ratio = self.tableau[i, 0] / a_ij
                if ratio < min_ratio - 1e-10:
                    min_ratio = ratio
                    candidates = [i]
                elif abs(ratio - min_ratio) <= 1e-10:
                    candidates.append(i)

        if not candidates:
            return -1

        if len(candidates) == 1:
            return candidates[0]

        # In caso di parità, la regola di Bland sceglie la variabile di base di indice minimo
        return min(candidates, key=lambda r: self.basis[r])

    def _pivot(self, pivot_row, pivot_col):
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_element
        for i in range(self.m + 1):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                self.tableau[i, :] -= factor * self.tableau[pivot_row, :]
        self.basis[pivot_row] = pivot_col - 1
        # Zero-snapping: azzera il rumore numerico fluttuante a valle delle operazioni elementari di riga
        self.tableau[np.abs(self.tableau) < 1e-12] = 0.0

    # ============================================================
    #  Risolutore Completo (Due Fasi)
    # ============================================================

    def solve(self, verbose=True, plot3d=True, save_figures=False, output_dir=None):
        if output_dir is None:
            output_dir = "."

        if verbose:
            print("╔══════════════════════════════════════════════════════════════════╗")
            print("║           METODO DEL SIMPLESSO – Risolutore PL                 ║")
            print("╠══════════════════════════════════════════════════════════════════╣")
            print("║  Problema formulato:                                           ║")
            obj_label_text = "min" if getattr(self, 'is_min', False) else "max"
            c_disp = -self.c_orig if getattr(self, 'is_min', False) else self.c_orig
            obj_str = " + ".join(f"{self._format_val(c_disp[i])}·{self.var_names[i]}"
                                 for i in range(self.n))
            print(f"║  {obj_label_text} z = {obj_str}")
            print("║  s.t.")
            for i in range(self.m):
                constr = " + ".join(f"{self._format_val(self.A_orig[i, j])}·{self.var_names[j]}"
                                    for j in range(self.n))
                s_char = '≤' if self.signs_orig[i] == '<=' else ('≥' if self.signs_orig[i] == '>=' else '=')
                print(f"║       {constr} {s_char} {self._format_val(self.b_orig[i])}")
            var_str = ", ".join(self.var_names)
            print(f"║       {var_str} ≥ 0")

            if self.normalization_notes:
                print("╟──────────────────────────────────────────────────────────────────╢")
                print("║  Normalizzazioni automatiche applicate:                        ║")
                for note in self.normalization_notes:
                    print(f"║   • {note}")

            print("╚══════════════════════════════════════════════════════════════════╝")

        iteration = 0

        # ── FASE 1 (se necessaria) ──
        if self.needs_phase1:
            if verbose:
                print("\n" + "▶" * 40)
                print("  FASE 1: Ricerca di una base ammissibile iniziale")
                print("▶" * 40)
                self.print_tableau(iteration=0, phase_name="Fase 1 Iniziale")

            art_start = 1 + self.n + len(self.slack_names)

            while True:
                # Cerca variabile entrante solo tra decisionali e slack/surplus
                pivot_col = self._find_pivot_column(max_col=art_start)
                if pivot_col == -1:
                    break

                pivot_row = self._find_pivot_row(pivot_col)
                if pivot_row == -1:
                    if verbose:
                        print("  [AVVISO] Problema illimitato in Fase 1")
                    self.status = 'unbounded'
                    self._save_state("Problema illimitato in Fase 1", phase=1, status='unbounded')
                    return None

                entering = self.all_var_names[pivot_col - 1]
                leaving = self.all_var_names[self.basis[pivot_row]]
                pivot_val = self.tableau[pivot_row, pivot_col]

                self._pivot(pivot_row, pivot_col)
                iteration += 1

                if iteration > 500:
                    if verbose:
                        print("  [ERRORE] Raggiunto il limite massimo di iterazioni in Fase 1.")
                    self.status = 'max_iterations'
                    self._save_state("Limite iterazioni raggiunto in Fase 1 (possibile ciclo)", phase=1, status='max_iterations')
                    return None

                pinfo = {
                    'entering': entering, 'leaving': leaving,
                    'pivot_row': pivot_row, 'pivot_col': pivot_col,
                    'pivot_value': pivot_val
                }
                self._save_state(f"Fase 1 – Iterazione {iteration}: {entering} entra, {leaving} esce",
                                 pivot_info=pinfo, phase=1)

                if verbose:
                    self.print_tableau(iteration=iteration, phase_name="Fase 1")

            # Verifica ammissibilità
            w_final = self.tableau[self.m, 0]
            if w_final > 1e-6:
                if verbose:
                    print("\n  [ERRORE] Problema inammissibile: l'insieme dei vincoli non ammette soluzioni (w > 0).")
                self.status = 'infeasible'
                self._save_state("Problema Inammissibile (nessuna soluzione)", phase=1, status='infeasible')
                return None

            if verbose:
                print("\n  [OK] Base ammissibile trovata alla fine della Fase 1")

            # Rimuovi variabili artificiali ancora eventualmente in base a livello zero
            # Se una riga i ha ancora una variabile artificiale in base:
            # cerca di scambiarla con una variabile originale/slack non in base con coefficiente non nullo.
            # Se tutti i coefficienti della riga sono 0 (vincolo ridondante 0=0), la riga viene azzerata/rimossa.
            redundant_rows = []
            for i in range(self.m):
                if self.basis[i] >= self.n + len(self.slack_names):
                    drivable = False
                    for j in range(1, art_start):
                        var_idx = j - 1
                        if var_idx not in self.basis and abs(self.tableau[i, j]) > 1e-10:
                            self._pivot(i, j)
                            drivable = True
                            break
                    if not drivable:
                        # Vincolo ridondante 0=0: assegna una variabile fittizia già non artificiale
                        # o azzera la variabile artificiale in base
                        redundant_rows.append(i)

            # Elimina colonne artificiali dal tableau
            self.tableau = np.delete(self.tableau, range(art_start, self.tableau.shape[1]), axis=1)
            self.all_var_names = self.var_names + self.slack_names

            # Per eventuali righe ridondanti in cui l'artificiale era rimasta in base:
            # trova un indice valido tra le variabili non in base
            for r in redundant_rows:
                for v in range(len(self.all_var_names)):
                    if v not in self.basis:
                        self.basis[r] = v
                        break

            # Configura riga obiettivo per Fase 2
            self.current_obj_label = "-z"
            self.tableau[self.m, 0] = 0.0
            self.tableau[self.m, 1:1+self.n] = self.c_orig
            self.tableau[self.m, 1+self.n:] = 0.0

            # Canonicalizza rispetto alla base corrente
            for i in range(self.m):
                b_var = self.basis[i]
                if 1 + b_var < self.tableau.shape[1]:
                    coeff = self.tableau[self.m, 1 + b_var]
                    if abs(coeff) > 1e-10:
                        self.tableau[self.m, :] -= coeff * self.tableau[i, :]

            self._save_state("Inizio Fase 2 (Base ammissibile)", phase=2)
            if verbose:
                print("\n" + "─" * 40)
                print("  FASE 2: Ottimizzazione della funzione obiettivo")
                print("─" * 40)
                self.print_tableau(iteration=iteration, phase_name="Inizio Fase 2")

        # ── FASE 2 (Ottimizzazione) ──
        while True:
            pivot_col = self._find_pivot_column()
            if pivot_col == -1:
                self.status = 'optimal'
                if self.history:
                    self.history[-1]['status'] = 'optimal'
                if verbose:
                    print("\n  [OTTIMO] Soluzione ottima trovata")
                    dec_vars, z = self.get_decision_variables()
                    sol_parts = [f"{self.var_names[i]} = {self._format_val(dec_vars[i])}"
                                 for i in range(self.n)]
                    print(f"   Soluzione: {', '.join(sol_parts)}")
                    print(f"   Valore ottimo z* = {self._format_val(z)}")
                break

            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row == -1:
                if verbose:
                    print("  [AVVISO] Problema illimitato")
                self.status = 'unbounded'
                self._save_state("Problema illimitato (funzione obiettivo non superiormente limitata)", phase=2, status='unbounded')
                return None

            entering = self.all_var_names[pivot_col - 1]
            leaving = self.all_var_names[self.basis[pivot_row]]
            pivot_val = self.tableau[pivot_row, pivot_col]

            self._pivot(pivot_row, pivot_col)
            iteration += 1

            if iteration > 500:
                if verbose:
                    print("  [ERRORE] Raggiunto il limite massimo di iterazioni in Fase 2.")
                self.status = 'max_iterations'
                self._save_state("Limite iterazioni raggiunto in Fase 2 (possibile ciclo)", phase=2, status='max_iterations')
                return None

            pinfo = {
                'entering': entering, 'leaving': leaving,
                'pivot_row': pivot_row, 'pivot_col': pivot_col,
                'pivot_value': pivot_val
            }
            self._save_state(f"Iterazione {iteration}: {entering} entra, {leaving} esce",
                             pivot_info=pinfo, phase=2)

            if verbose:
                self.print_tableau(iteration=iteration, phase_name="Fase 2" if self.needs_phase1 else None)

        if plot3d and self.n == 3:
            self._plot_3d_feasible_region(save_figures, output_dir)
        elif plot3d and self.n == 2:
            self._plot_2d_feasible_region(save_figures, output_dir)

        dec_vars, z = self.get_decision_variables()
        return dec_vars, z

    # ============================================================
    #  Teoria della Dualità
    # ============================================================

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
        """
        Calcola il vettore dei moltiplicatori duali y all'iterazione step_idx.
        Risolve il sistema B^T pi = c_B corrispondente alla base attiva a quello step.
        Funziona esattamente e robustamente per qualsiasi combinazione di vincoli (<=, >=, =).
        """
        if not self.history or step_idx < 0 or step_idx >= len(self.history):
            return None

        state = self.history[step_idx]
        basis = state['basis']
        m, n = self.m, self.n

        # Se siamo in Fase 1, non siamo ancora in una base ammissibile della Fase 2
        if state.get('phase', 2) == 1:
            return np.zeros(m)

        # Matrice A_std per la Fase 2: decisionali (A_norm) + colonne di slack/surplus
        n_slack = len(self.slack_names)
        A_std = np.zeros((m, n + n_slack))
        A_std[:, :n] = self.A_norm
        for k, (row_idx, coeff) in enumerate(self.slack_cols):
            A_std[row_idx, n + k] = coeff

        c_std = np.zeros(n + n_slack)
        c_std[:n] = self.c_orig

        # Variabili valide in base (escludendo eventuali artificiali residue di Fase 1)
        valid_basis = [b for b in basis if b < n + n_slack]
        if len(valid_basis) != m:
            return np.zeros(m)

        B = A_std[:, valid_basis]
        cB = c_std[valid_basis]

        try:
            pi, _, _, _ = np.linalg.lstsq(B.T, cB, rcond=1e-10)
        except Exception:
            return np.zeros(m)

        # Aggiusta per l'eventuale inversione di segno di b_orig < 0
        y = np.zeros(m)
        for i in range(m):
            if self.b_orig[i] < -1e-10:
                y[i] = -pi[i]
            else:
                y[i] = pi[i]

        # Il solver lavora sempre in forma di massimo (c negato se il problema era di min):
        # per un problema di minimo il duale del problema originale ha segno opposto.
        if getattr(self, 'is_min', False):
            y = -y

        y[np.abs(y) < 1e-10] = 0.0
        return y

    def get_dual_formulation(self, is_primal_min=False):
        """
        Restituisce la formulazione esatta del problema duale associato.
        Ritorna: obj_type, b_coeff, A_dual, c_bounds, y_signs
        """
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
        if any(xj < -1e-7 for xj in x):
            return False
        for i in range(self.m):
            val = np.dot(self.A_orig[i], x)
            s = self.signs_orig[i]
            bi = self.b_orig[i]
            if s == '<=' and val > bi + 1e-7:
                return False
            elif s == '>=' and val < bi - 1e-7:
                return False
            elif s == '=' and abs(val - bi) > 1e-7:
                return False
        return True

    def find_all_vertices(self):
        """Trova tutti i vertici ammissibili della regione del poliedro."""
        n = self.n
        A_all = np.vstack([self.A_orig, np.eye(n)])
        b_all = np.hstack([self.b_orig, np.zeros(n)])
        total_planes = len(b_all)
        vertices = []

        for combo in combinations(range(total_planes), n):
            A_sub = A_all[list(combo)]
            b_sub = b_all[list(combo)]
            try:
                if abs(np.linalg.det(A_sub)) < 1e-10:
                    continue
                x = np.linalg.solve(A_sub, b_sub)
                if self._is_point_feasible(x):
                    x = np.maximum(x, 0.0)
                    if not any(np.allclose(x, v, atol=1e-6) for v in vertices):
                        vertices.append(x)
            except np.linalg.LinAlgError:
                continue

        return np.array(vertices) if vertices else np.array([]).reshape(0, n)

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
        verts = self.find_all_vertices()
        if len(verts) >= 4:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(verts)
                faces = [[verts[i] for i in simplex] for simplex in hull.simplices]
                poly = Poly3DCollection(faces, alpha=alpha, facecolor='cyan',
                                        edgecolor='navy', linewidth=0.5)
                ax.add_collection3d(poly)
                ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                           c='navy', s=25, zorder=5, alpha=0.6)
            except Exception:
                ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                           c='navy', s=25, zorder=5, alpha=0.6)
        elif len(verts) > 0:
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                       c='navy', s=25, zorder=5, alpha=0.6)

        if len(verts) > 0:
            margin = 0.5
            ax.set_xlim([0, max(verts[:, 0].max(), 0.1) + margin])
            ax.set_ylim([0, max(verts[:, 1].max(), 0.1) + margin])
            ax.set_zlim([0, max(verts[:, 2].max(), 0.1) + margin])

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
                   edgecolors='darkorange', linewidths=2, label='Ottimo')
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
            if abs(a2) > 1e-10:
                y_line = (bi - a1 * x_range) / a2
                lbl = f"{self._format_val(a1)}{self.var_names[0]}+{self._format_val(a2)}{self.var_names[1]} {s} {self._format_val(bi)}"
                ax.plot(x_range, y_line, label=lbl, linewidth=1.5)
            elif abs(a1) > 1e-10:
                ax.axvline(x=bi/a1, label=f"{self.var_names[0]} {s} {self._format_val(bi/a1)}", linewidth=1.5)

        verts = self.find_all_vertices()
        if len(verts) >= 3:
            from matplotlib.patches import Polygon
            center = np.mean(verts, axis=0)
            angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
            sorted_verts = verts[np.argsort(angles)]
            poly = Polygon(sorted_verts, alpha=alpha, facecolor='lightblue', edgecolor='navy')
            ax.add_patch(poly)

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
                   edgecolors='darkorange', linewidths=2, label='Ottimo')
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
