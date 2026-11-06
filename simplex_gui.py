#!/usr/bin/env python3
"""
SimplexStudio – Interfaccia Grafica Interattiva
================================================
GUI con:
  - Menu a tendina per variabili, vincoli, segno dei vincoli, min/max
  - Tableau del simplesso aggiornato passo-passo
  - Visualizzazione 2D / 3D della regione ammissibile
  - Animazione del percorso del simplesso vertice per vertice
  - Controlli: Avanti, Indietro, Play, Velocita, Reset

Copyright (c) 2026 Alessandro Carboni
carboni.alessandro@outlook.it

Dipendenze: numpy, matplotlib, scipy (opzionale), tkinter
"""

import sys
import os

# Imposta cache scrivibile per Matplotlib
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from fractions import Fraction
import platform

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Importa il solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simplex import SimplexSolver

# ═══════════════════════════════════════════════════════
#  Costanti di stile
# ═══════════════════════════════════════════════════════

IS_MAC = platform.system() == 'Darwin'
FONT_MONO = ('Menlo', 13) if IS_MAC else ('Consolas', 11)
FONT_MONO_SM = ('Menlo', 11) if IS_MAC else ('Consolas', 10)
FONT_UI = ('Helvetica Neue', 13) if IS_MAC else ('Segoe UI', 11)
FONT_UI_BOLD = ('Helvetica Neue', 13, 'bold') if IS_MAC else ('Segoe UI', 11, 'bold')
FONT_TITLE = ('Helvetica Neue', 18, 'bold') if IS_MAC else ('Segoe UI', 15, 'bold')
FONT_SECTION = ('Helvetica Neue', 14, 'bold') if IS_MAC else ('Segoe UI', 12, 'bold')

COL = {
    'bg':             '#ECEFF1',
    'card':           '#FFFFFF',
    'primary':        '#1565C0',
    'prim_fg':        '#FFFFFF',
    'success':        '#2E7D32',
    'succ_lt':        '#C8E6C9',
    'warning':        '#EF6C00',
    'danger':         '#C62828',
    'text':           '#263238',
    'text2':          '#546E7A',
    'border':         '#B0BEC5',
    'pivot_bg':       '#FFF176',
    'header_bg':      '#E3F2FD',
    'sep':            '#90A4AE',
    'vertex_start':   '#2E7D32',  # Verde per start
    'vertex_optimal': '#FFD700',  # Oro per ottimo
    'vertex_current': '#E53935',  # Rosso per iterazione corrente
}

# ═══════════════════════════════════════════════════════
#  Applicazione GUI
# ═══════════════════════════════════════════════════════

class SimplexGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("SimplexStudio – Programmazione Lineare")
        self.root.configure(bg=COL['bg'])

        # Dimensione finestra
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1440, sw - 80), min(920, sh - 80)
        root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        root.minsize(1100, 700)

        # ── Stato ──
        self.n_vars = tk.IntVar(value=3)
        self.n_constr = tk.IntVar(value=3)
        self.obj_type = tk.StringVar(value='max')
        self.speed_ms = tk.IntVar(value=1200)

        self.solver = None
        self.current_step = 0
        self.total_steps = 0
        self.anim_id = None

        # Widget containers per form dinamico
        self.obj_entries = []
        self.constr_rows = []          # lista di dict per ogni vincolo

        # ── Costruisci UI ──
        self._build_ui()
        self._rebuild_form()

        # ── Shortcuts ──
        root.bind('<Right>',  lambda e: self._step_forward())
        root.bind('<Left>',   lambda e: self._step_backward())
        root.bind('<space>',  lambda e: self._toggle_animation())
        root.bind('<Escape>', lambda e: self._stop_animation())

    # ═══════════════════════════════════════════════════
    #  Costruzione UI
    # ═══════════════════════════════════════════════════

    def _build_ui(self):
        # Paned window principale (sinistra / destra)
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                     bg=COL['border'], sashwidth=4,
                                     sashrelief=tk.FLAT)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── Pannello sinistro (input) ──
        left = tk.Frame(self.paned, bg=COL['bg'], width=450)
        self.paned.add(left, minsize=400, width=450)
        self._build_left(left)

        # ── Pannello destro (visualizzazione) ──
        right = tk.Frame(self.paned, bg=COL['bg'])
        self.paned.add(right, minsize=600)
        self._build_right(right)

    # ─── Pannello sinistro ────────────────────────────

    def _build_left(self, parent):
        canvas = tk.Canvas(parent, bg=COL['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        self.left_inner = tk.Frame(canvas, bg=COL['bg'])

        self.left_inner.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.left_inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scroll con mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")
        canvas.bind_all('<MouseWheel>', _on_mousewheel)

        f = self.left_inner
        pad = {'padx': 10, 'pady': 4}

        # ── Titolo ──
        tk.Label(f, text="Configurazione", font=FONT_TITLE,
                 bg=COL['bg'], fg=COL['primary']).pack(anchor='w', padx=10, pady=(10, 8))

        # ── Configurazione ──
        cfg = tk.LabelFrame(f, text=" Problema ", font=FONT_SECTION,
                            bg=COL['card'], fg=COL['text'], bd=1,
                            relief=tk.GROOVE, padx=12, pady=8)
        cfg.pack(fill=tk.X, **pad)

        r = 0
        tk.Label(cfg, text="N° variabili:", font=FONT_UI, bg=COL['card']).grid(
            row=r, column=0, sticky='w', pady=3)
        vars_cb = ttk.Combobox(cfg, textvariable=self.n_vars, values=[2, 3, 4, 5, 6],
                               width=5, state='readonly', font=FONT_UI)
        vars_cb.grid(row=r, column=1, sticky='w', padx=(8, 0), pady=3)
        vars_cb.bind('<<ComboboxSelected>>', lambda e: self._rebuild_form())

        r += 1
        tk.Label(cfg, text="N° vincoli:", font=FONT_UI, bg=COL['card']).grid(
            row=r, column=0, sticky='w', pady=3)
        constr_cb = ttk.Combobox(cfg, textvariable=self.n_constr,
                                 values=list(range(1, 11)),
                                 width=5, state='readonly', font=FONT_UI)
        constr_cb.grid(row=r, column=1, sticky='w', padx=(8, 0), pady=3)
        constr_cb.bind('<<ComboboxSelected>>', lambda e: self._rebuild_form())

        r += 1
        tk.Label(cfg, text="Obiettivo:", font=FONT_UI, bg=COL['card']).grid(
            row=r, column=0, sticky='w', pady=3)
        obj_cb = ttk.Combobox(cfg, textvariable=self.obj_type,
                              values=['max', 'min'], width=5,
                              state='readonly', font=FONT_UI)
        obj_cb.grid(row=r, column=1, sticky='w', padx=(8, 0), pady=3)

        # ── Funzione obiettivo ──
        self.obj_frame = tk.LabelFrame(f, text=" Funzione obiettivo ",
                                        font=FONT_SECTION, bg=COL['card'],
                                        fg=COL['text'], bd=1, relief=tk.GROOVE,
                                        padx=12, pady=8)
        self.obj_frame.pack(fill=tk.X, **pad)

        # ── Vincoli ──
        self.constr_frame = tk.LabelFrame(f, text=" Vincoli ", font=FONT_SECTION,
                                           bg=COL['card'], fg=COL['text'], bd=1,
                                           relief=tk.GROOVE, padx=12, pady=8)
        self.constr_frame.pack(fill=tk.X, **pad)

        # ── Esempi ──
        ex_frame = tk.LabelFrame(f, text=" Esempi ", font=FONT_SECTION,
                                  bg=COL['card'], fg=COL['text'], bd=1,
                                  relief=tk.GROOVE, padx=12, pady=8)
        ex_frame.pack(fill=tk.X, **pad)

        tk.Button(ex_frame, text="Esempio in R³",
                  font=FONT_UI, bg='#E3F2FD', fg=COL['primary'],
                  activebackground='#BBDEFB', relief=tk.FLAT, cursor='hand2',
                  command=lambda: self._load_example('3var')
                  ).pack(fill=tk.X, pady=2)
        tk.Button(ex_frame, text="Esempio in R²",
                  font=FONT_UI, bg='#E8F5E9', fg=COL['success'],
                  activebackground='#C8E6C9', relief=tk.FLAT, cursor='hand2',
                  command=lambda: self._load_example('2var')
                  ).pack(fill=tk.X, pady=2)

        # ── Pulsanti azione ──
        btn_frame = tk.Frame(f, bg=COL['bg'])
        btn_frame.pack(fill=tk.X, padx=10, pady=(12, 10))

        # Pulsante RISOLVI: realizzato con tk.Label per garantire il colore rosso vivo anche su macOS
        self.solve_btn = tk.Label(btn_frame, text="RISOLVI",
                                  font=(FONT_UI_BOLD[0], FONT_UI_BOLD[1] + 2, 'bold'),
                                  bg='#D32F2F', fg='#FFFFFF',
                                  cursor='hand2', pady=10, relief=tk.RAISED, bd=1)
        self.solve_btn.pack(fill=tk.X, pady=4)
        self.solve_btn.bind('<Button-1>', lambda e: self._solve())
        self.solve_btn.bind('<Enter>', lambda e: self.solve_btn.config(bg='#B71C1C'))
        self.solve_btn.bind('<Leave>', lambda e: self.solve_btn.config(bg='#D32F2F'))

        # Pulsante Reset
        self.reset_btn = tk.Label(btn_frame, text="Reset", font=FONT_UI,
                                  bg='#ECEFF1', fg=COL['text2'],
                                  cursor='hand2', pady=6, relief=tk.GROOVE, bd=1)
        self.reset_btn.pack(fill=tk.X, pady=4)
        self.reset_btn.bind('<Button-1>', lambda e: self._reset())
        self.reset_btn.bind('<Enter>', lambda e: self.reset_btn.config(bg='#CFD8DC'))
        self.reset_btn.bind('<Leave>', lambda e: self.reset_btn.config(bg='#ECEFF1'))

        # ── Copyright ──
        copy_frame = tk.Frame(f, bg=COL['bg'])
        copy_frame.pack(fill=tk.X, padx=10, pady=(20, 10))
        FONT_COPY = ('Helvetica Neue', 10) if IS_MAC else ('Segoe UI', 9)
        tk.Label(copy_frame,
                 text="© 2026 Alessandro Carboni\ncarboni.alessandro@outlook.it",
                 font=FONT_COPY, bg=COL['bg'], fg=COL['text2'],
                 justify=tk.CENTER).pack(anchor='center')

    # ─── Pannello destro ──────────────────────────────

    def _build_right(self, parent):
        # Paned verticale: plot in alto, tableau in basso
        vpaned = tk.PanedWindow(parent, orient=tk.VERTICAL,
                                 bg=COL['border'], sashwidth=4)
        vpaned.pack(fill=tk.BOTH, expand=True)

        # ── Plot ──
        plot_frame = tk.Frame(vpaned, bg=COL['card'], bd=1, relief=tk.GROOVE)
        vpaned.add(plot_frame, minsize=250, height=450)

        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor=COL['card'])
        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Parte bassa: controlli + tableau ──
        bottom = tk.Frame(vpaned, bg=COL['bg'])
        vpaned.add(bottom, minsize=200)

        # Controlli
        ctrl = tk.Frame(bottom, bg=COL['card'], bd=1, relief=tk.GROOVE, padx=10, pady=6)
        ctrl.pack(fill=tk.X, padx=2, pady=(2, 4))

        self.step_label = tk.Label(ctrl, text="Premi RISOLVI per iniziare",
                                    font=FONT_UI_BOLD, bg=COL['card'], fg=COL['text'])
        self.step_label.pack(side=tk.LEFT, padx=(0, 15))

        tk.Button(ctrl, text="Indietro", font=FONT_UI, bg='#E0E0E0',
                  relief=tk.FLAT, cursor='hand2',
                  command=self._step_backward).pack(side=tk.LEFT, padx=3)

        tk.Button(ctrl, text="Avanti", font=FONT_UI, bg='#E0E0E0',
                  relief=tk.FLAT, cursor='hand2',
                  command=self._step_forward).pack(side=tk.LEFT, padx=3)

        self.play_btn = tk.Button(ctrl, text="Play", font=FONT_UI_BOLD,
                                   bg=COL['succ_lt'], fg=COL['success'],
                                   relief=tk.FLAT, cursor='hand2',
                                   command=self._toggle_animation)
        self.play_btn.pack(side=tk.LEFT, padx=(12, 3))

        tk.Label(ctrl, text="Velocità:", font=FONT_UI, bg=COL['card']).pack(
            side=tk.LEFT, padx=(15, 3))
        speed_scale = ttk.Scale(ctrl, from_=200, to=3000, variable=self.speed_ms,
                                 orient=tk.HORIZONTAL, length=120)
        speed_scale.pack(side=tk.LEFT, padx=3)
        self.speed_val_label = tk.Label(ctrl, text="1.2s", font=FONT_MONO_SM,
                                         bg=COL['card'], fg=COL['text2'], width=5)
        self.speed_val_label.pack(side=tk.LEFT)
        self.speed_ms.trace_add('write', self._update_speed_label)

        # Tableau (Text widget)
        tab_frame = tk.Frame(bottom, bg=COL['card'], bd=1, relief=tk.GROOVE)
        tab_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        tk.Label(tab_frame, text=" Tableau del Simplesso", font=FONT_SECTION,
                 bg=COL['header_bg'], fg=COL['primary'], anchor='w'
                 ).pack(fill=tk.X, padx=1, pady=(1, 0))

        self.tableau_text = tk.Text(tab_frame, font=FONT_MONO, bg=COL['card'],
                                     fg=COL['text'], wrap=tk.NONE,
                                     relief=tk.FLAT, padx=10, pady=8,
                                     state=tk.DISABLED, height=10)
        t_scroll_x = ttk.Scrollbar(tab_frame, orient=tk.HORIZONTAL,
                                    command=self.tableau_text.xview)
        t_scroll_y = ttk.Scrollbar(tab_frame, orient=tk.VERTICAL,
                                    command=self.tableau_text.yview)
        self.tableau_text.configure(xscrollcommand=t_scroll_x.set,
                                     yscrollcommand=t_scroll_y.set)
        t_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        t_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tableau_text.pack(fill=tk.BOTH, expand=True)

        # Tag di stile per il tableau
        self.tableau_text.tag_configure('header', foreground=COL['primary'],
                                         font=(*FONT_MONO[:1], FONT_MONO[1], 'bold'))
        self.tableau_text.tag_configure('separator', foreground=COL['sep'])
        self.tableau_text.tag_configure('pivot', background=COL['pivot_bg'],
                                         font=(*FONT_MONO[:1], FONT_MONO[1], 'bold'))
        self.tableau_text.tag_configure('info', foreground=COL['text2'],
                                         font=FONT_UI)
        self.tableau_text.tag_configure('optimal', foreground=COL['success'],
                                         font=(*FONT_MONO[:1], FONT_MONO[1], 'bold'))
        self.tableau_text.tag_configure('entering', foreground=COL['success'])
        self.tableau_text.tag_configure('leaving', foreground=COL['danger'])
        self.tableau_text.tag_configure('normal', foreground=COL['text'])

    # ═══════════════════════════════════════════════════
    #  Form dinamico
    # ═══════════════════════════════════════════════════

    def _rebuild_form(self):
        """Ricostruisce i campi di input in base a n_vars e n_constr."""
        n = self.n_vars.get()
        m = self.n_constr.get()

        # ── Funzione obiettivo ──
        for w in self.obj_frame.winfo_children():
            w.destroy()
        self.obj_entries = []

        row = tk.Frame(self.obj_frame, bg=COL['card'])
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="z =", font=FONT_UI_BOLD, bg=COL['card']).pack(
            side=tk.LEFT, padx=(0, 5))

        # Larghezza entry adattiva al numero di variabili
        entry_w = 5 if n <= 3 else (4 if n <= 5 else 3)

        for i in range(n):
            if i > 0:
                tk.Label(row, text=" + ", font=FONT_UI, bg=COL['card']).pack(side=tk.LEFT)
            e = tk.Entry(row, width=entry_w, font=FONT_MONO, justify=tk.CENTER,
                         relief=tk.SOLID, bd=1)
            e.pack(side=tk.LEFT, padx=1)
            e.insert(0, "0")
            self.obj_entries.append(e)
            tk.Label(row, text=f"x{i+1}", font=FONT_UI, bg=COL['card'],
                     fg=COL['primary']).pack(side=tk.LEFT, padx=(1, 3))

        # ── Vincoli ──
        for w in self.constr_frame.winfo_children():
            w.destroy()
        self.constr_rows = []

        for i in range(m):
            self._add_constraint_row(i, n)

    def _add_constraint_row(self, idx, n):
        """Aggiunge una riga vincolo al form."""
        row_frame = tk.Frame(self.constr_frame, bg=COL['card'])
        row_frame.pack(fill=tk.X, pady=3)

        tk.Label(row_frame, text=f"V{idx+1}:", font=FONT_UI,
                 bg=COL['card'], fg=COL['text2'], width=3).pack(side=tk.LEFT)

        # Larghezza entry adattiva al numero di variabili
        entry_w = 5 if n <= 3 else (4 if n <= 5 else 3)

        coeff_entries = []
        for j in range(n):
            if j > 0:
                tk.Label(row_frame, text="+", font=FONT_UI,
                         bg=COL['card'], fg=COL['text2']).pack(side=tk.LEFT, padx=1)
            e = tk.Entry(row_frame, width=entry_w, font=FONT_MONO, justify=tk.CENTER,
                         relief=tk.SOLID, bd=1)
            e.pack(side=tk.LEFT, padx=1)
            e.insert(0, "0")
            coeff_entries.append(e)
            tk.Label(row_frame, text=f"x{j+1}", font=FONT_UI, bg=COL['card'],
                     fg=COL['primary']).pack(side=tk.LEFT, padx=(0, 2))

        # Segno del vincolo
        sign_var = tk.StringVar(value='≤')
        sign_cb = ttk.Combobox(row_frame, textvariable=sign_var,
                                values=['≤', '≥', '='], width=3,
                                state='readonly', font=FONT_MONO)
        sign_cb.pack(side=tk.LEFT, padx=5)

        # Termine noto
        rhs_e = tk.Entry(row_frame, width=5, font=FONT_MONO, justify=tk.CENTER,
                         relief=tk.SOLID, bd=1)
        rhs_e.pack(side=tk.LEFT, padx=2)
        rhs_e.insert(0, "0")

        self.constr_rows.append({
            'frame': row_frame,
            'coeffs': coeff_entries,
            'sign': sign_var,
            'rhs': rhs_e
        })

    # ═══════════════════════════════════════════════════
    #  Carica esempi
    # ═══════════════════════════════════════════════════

    def _load_example(self, name):
        """Carica un esempio predefinito."""
        if name == '3var':
            # Esempio in R3: max z = 3x1 + 2x2 + 4x3
            # s.t. x1 + x2 + x3 <= 4, 2x1 + x2 + 3x3 <= 6, x1 + 2x2 + x3 <= 5
            self.n_vars.set(3)
            self.n_constr.set(3)
            self.obj_type.set('max')
            self._rebuild_form()
            c = [3, 2, 4]
            A = [
                [1, 1, 1],
                [2, 1, 3],
                [1, 2, 1]
            ]
            b = [4, 6, 5]
            signs = ['≤', '≤', '≤']
        elif name == '2var':
            # Esempio in R2: max z = x1 + 3x2
            # s.t. 6x1 + 10x2 <= 30, 3x1 + 2x2 >= 6, x1 - 2x2 >= -1, x2 >= 1/2
            self.n_vars.set(2)
            self.n_constr.set(4)
            self.obj_type.set('max')
            self._rebuild_form()
            c = [1, 3]
            A = [
                [6, 10],
                [3, 2],
                [1, -2],
                [0, 1]
            ]
            b = [30, 6, -1, 0.5]
            signs = ['≤', '≥', '≥', '≥']
        else:
            return

        # Compila i campi
        for i, e in enumerate(self.obj_entries):
            e.delete(0, tk.END)
            val = c[i]
            e.insert(0, str(Fraction(val).limit_denominator(1000)))

        for i, row in enumerate(self.constr_rows):
            for j, e in enumerate(row['coeffs']):
                e.delete(0, tk.END)
                val = A[i][j]
                e.insert(0, str(Fraction(val).limit_denominator(1000)))
            row['sign'].set(signs[i])
            row['rhs'].delete(0, tk.END)
            e_val = b[i]
            row['rhs'].insert(0, str(Fraction(e_val).limit_denominator(1000)))

    # ═══════════════════════════════════════════════════
    #  Parsing e validazione input
    # ═══════════════════════════════════════════════════

    def _parse_float(self, text):
        """Converte testo in float. Supporta frazioni come 1/2."""
        text = text.strip()
        if not text:
            return 0.0
        if '/' in text:
            parts = text.split('/')
            return float(Fraction(int(parts[0]), int(parts[1])))
        return float(text)

    def _parse_inputs(self):
        """Legge i valori dai campi e restituisce c, A, b, signs, is_min."""
        n = self.n_vars.get()

        # Obiettivo
        c = []
        for e in self.obj_entries:
            c.append(self._parse_float(e.get()))

        # Se min, converti in max negando c
        is_min = self.obj_type.get() == 'min'
        if is_min:
            c = [-ci for ci in c]

        # Vincoli originali con i loro segni
        A = []
        b = []
        signs = []
        for row in self.constr_rows:
            coeffs = [self._parse_float(e.get()) for e in row['coeffs']]
            rhs = self._parse_float(row['rhs'].get())
            s = row['sign'].get()
            s_clean = '<=' if s in ('<=', '≤') else ('>=' if s in ('>=', '≥') else '=')

            A.append(coeffs)
            b.append(rhs)
            signs.append(s_clean)

        return c, A, b, signs, is_min

    # ═══════════════════════════════════════════════════
    #  Risoluzione
    # ═══════════════════════════════════════════════════

    def _solve(self):
        """Risolve il problema e inizializza la visualizzazione."""
        self._stop_animation()

        try:
            c, A, b, signs, is_min = self._parse_inputs()
        except (ValueError, ZeroDivisionError) as e:
            messagebox.showerror("Errore nei dati",
                f"Controlla i valori inseriti.\nDettaglio: {e}")
            return

        # Crea il solver con gestione automatica di segni, b<0 e Due Fasi
        n_orig = self.n_vars.get()
        var_names = [f"x{i+1}" for i in range(n_orig)]
        try:
            self.solver = SimplexSolver(c, A, b, signs=signs, var_names=var_names)
            result = self.solver.solve(verbose=False, plot3d=False)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la risoluzione:\n{e}")
            return

        self.is_min = is_min
        self.total_steps = len(self.solver.history)
        self.current_step = 0

        if result is None:
            last_desc = self.solver.history[-1]['description'] if self.solver.history else ""
            if "Inammissibile" in last_desc:
                messagebox.showwarning("Inammissibile",
                    "L'insieme dei vincoli non ammette soluzioni ammissibili (regione ammissibile vuota).")
            else:
                messagebox.showwarning("Illimitato",
                    "Il problema è illimitato (la funzione obiettivo può crescere all'infinito).")

        # Mostra il primo passo
        self._display_step(0)

    # ═══════════════════════════════════════════════════
    #  Visualizzazione di un passo
    # ═══════════════════════════════════════════════════

    def _display_step(self, step):
        """Aggiorna tableau e grafico per il passo indicato."""
        if not self.solver or step < 0 or step >= self.total_steps:
            return

        self.current_step = step
        state = self.solver.history[step]
        phase = state.get('phase', 2)
        
        # Se il problema non necessita di Due Fasi, non mostriamo "Fase 2" per non confondere
        if self.solver.needs_phase1:
            phase_prefix = "Fase 1 (Ricerca Soluzione di Base Ammissibile)" if phase == 1 else "Fase 2 (Ottimizzazione)"
            prefix = f"{phase_prefix} - "
        else:
            prefix = ""

        # Aggiorna label
        if step == self.total_steps - 1:
            if "Inammissibile" in state['description']:
                self.step_label.config(
                    text="Inammissibile: nessuna soluzione possibile",
                    fg=COL['danger'])
            else:
                self.step_label.config(
                    text=f"{prefix}Passo {step}/{self.total_steps-1} (OTTIMO)",
                    fg=COL['success'])
        else:
            self.step_label.config(
                text=f"{prefix}Passo {step}/{self.total_steps-1}",
                fg=COL['text'])

        self._render_tableau(state, step)
        self._render_plot(step)

    # ─── Rendering del tableau ────────────────────────

    def _frac(self, val):
        """Formatta un valore come stringa di frazione."""
        return str(Fraction(val).limit_denominator(1000))

    def _render_tableau(self, state, step):
        """Genera il tableau nel widget Text con colori."""
        t = self.tableau_text
        t.config(state=tk.NORMAL)
        t.delete('1.0', tk.END)

        tableau = state['tableau']
        basis = state['basis']
        m = self.solver.m
        names = state['all_var_names']
        n_total = len(names)
        obj_lbl = state.get('obj_label', '-z')

        # Pivot info per highlight
        pivot_info = state.get('pivot_info')
        pivot_rc = None
        if pivot_info:
            pivot_rc = (pivot_info['pivot_row'], pivot_info['pivot_col'])

        # Calcola larghezze colonne
        col_widths = []
        for k in range(n_total):
            w = max(7, len(names[k]) + 2)
            for i in range(m + 1):
                w = max(w, len(self._frac(tableau[i, 1 + k])) + 2)
            col_widths.append(w)
        z_w = max(7, len(obj_lbl) + 2)
        for i in range(m + 1):
            z_w = max(z_w, len(self._frac(tableau[i, 0])) + 2)

        margin = "   "

        # ── Header ──
        header = margin
        for k in range(n_total):
            header += f" {names[k]:^{col_widths[k]}}"
        header += f"  │ {obj_lbl:^{z_w}}\n"
        t.insert(tk.END, header, 'header')

        # ── Riga obiettivo (sopra la linea) ──
        obj_line = margin
        for k in range(n_total):
            val_str = self._frac(tableau[m, 1 + k])
            obj_line += f" {val_str:^{col_widths[k]}}"
        z_val = self._frac(tableau[m, 0])
        obj_line += f"  │ {z_val:^{z_w}}\n"
        t.insert(tk.END, obj_line, 'normal')

        # ── Separatore ──
        line_w = sum(col_widths) + n_total + len(margin)
        sep = margin + "─" * (line_w - len(margin)) + "──┼─" + "─" * z_w + "\n"
        t.insert(tk.END, sep, 'separator')

        # ── Righe vincoli ──
        for i in range(m):
            row_str = margin
            for k in range(n_total):
                val_str = self._frac(tableau[i, 1 + k])
                cell = f" {val_str:^{col_widths[k]}}"

                # Evidenzia pivot
                if pivot_rc and i == pivot_rc[0] and (1 + k) == pivot_rc[1]:
                    t.insert(tk.END, row_str)
                    row_str = ""
                    t.insert(tk.END, cell, 'pivot')
                    continue

                row_str += cell

            if row_str:
                t.insert(tk.END, row_str)

            b_val = self._frac(tableau[i, 0])
            t.insert(tk.END, f"  │ {b_val:^{z_w}}\n", 'normal')

        # ── Info sulla soluzione ──
        t.insert(tk.END, "\n")
        dec = state['decision_vars']
        z_val = state['z_value']
        if self.is_min and state.get('phase', 2) == 2:
            z_val = -z_val
        n_dec = self.solver.n
        sol_parts = [f"{self.solver.var_names[i]}={self._frac(dec[i])}"
                     for i in range(n_dec)]

        t.insert(tk.END, f"  Soluzione di Base Ammissibile: ({', '.join(sol_parts)})\n", 'info')

        lbl_display = obj_lbl.replace('-', '')
        z_display = self._frac(z_val)
        tag = 'optimal' if step == self.total_steps - 1 else 'info'
        t.insert(tk.END, f"  {lbl_display} = {z_display}", tag)

        if step == self.total_steps - 1:
            t.insert(tk.END, "  [OTTIMO]", 'optimal')

        t.insert(tk.END, "\n")
        basis_names = [names[b] for b in basis if b < len(names)]
        t.insert(tk.END, f"  Base: {{{', '.join(basis_names)}}}\n", 'info')

        # Info normalizzazioni applicate (al passo 0)
        if step == 0 and self.solver.normalization_notes:
            t.insert(tk.END, "\n  Normalizzazioni automatiche (b < 0):\n", 'info')
            for note in self.solver.normalization_notes:
                t.insert(tk.END, f"    - {note}\n", 'info')

        # Info pivot (per il passo successivo)
        if step < self.total_steps - 1:
            next_state = self.solver.history[step + 1]
            pi = next_state.get('pivot_info')
            if pi:
                t.insert(tk.END, f"\n  Prossimo pivot:\n", 'info')
                t.insert(tk.END, f"    Entra: {pi['entering']}\n", 'entering')
                t.insert(tk.END, f"    Esce:  {pi['leaving']}\n", 'leaving')

        t.config(state=tk.DISABLED)

    # ─── Rendering del grafico ────────────────────────

    def _render_plot(self, step):
        """Aggiorna il grafico matplotlib."""
        self.fig.clear()
        n = self.solver.n

        if n == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            self._plot_3d(ax, step)
        elif n == 2:
            ax = self.fig.add_subplot(111)
            self._plot_2d(ax, step)
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Visualizzazione non disponibile\nper {n} variabili",
                    ha='center', va='center', fontsize=14, color=COL['text2'],
                    transform=ax.transAxes)
            ax.set_axis_off()

        self.fig.tight_layout()
        self.canvas_mpl.draw()

    def _find_vertices(self):
        """Trova i vertici della regione ammissibile supportando qualsiasi vincolo (≤, ≥, =)."""
        if self.solver:
            return self.solver.find_all_vertices()
        return np.array([]).reshape(0, 2)

    def _plot_3d(self, ax, step):
        """Disegna la regione ammissibile 3D e il percorso del simplesso."""
        verts = self._find_vertices()
        names = self.solver.var_names

        if len(verts) >= 4 and HAS_SCIPY:
            hull = ConvexHull(verts)
            faces = [[verts[i] for i in simplex] for simplex in hull.simplices]
            poly = Poly3DCollection(faces, alpha=0.12, facecolor='cyan',
                                     edgecolor='navy', linewidth=0.5)
            ax.add_collection3d(poly)

        if len(verts) > 0:
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                       c='navy', s=20, alpha=0.4, zorder=5)

        # Percorso del simplesso fino allo step corrente
        points = []
        for s in range(step + 1):
            dec = self.solver.history[s]['decision_vars']
            if len(dec) >= 3:
                points.append(dec[:3])

        if points:
            pts = np.array(points)

            # Linee del percorso
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        'r-', linewidth=2.5, zorder=8, alpha=0.8)

            # Vertici visitati
            for i, pt in enumerate(pts):
                if i == 0:
                    color, marker, size = COL['vertex_start'], '^', 120
                    label = "Start"
                elif i == step and step == self.total_steps - 1:
                    color, marker, size = COL['vertex_optimal'], '*', 200
                    label = "Ottimo"
                elif i == step:
                    color, marker, size = COL['vertex_current'], 'D', 100
                    label = "Corrente"
                else:
                    color, marker, size = '#FF8A80', 'o', 60
                    label = None

                ax.scatter(*pt, c=color, s=size, marker=marker, zorder=10,
                           edgecolors='black', linewidths=1,
                           label=label if label else None)

                z_val = self.solver.history[i]['z_value']
                if self.is_min:
                    z_val = -z_val
                ax.text(pt[0] + 0.08, pt[1] + 0.08, pt[2] + 0.08,
                        f"It.{i}\nz={self._frac(z_val)}",
                        fontsize=8, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  alpha=0.85, edgecolor='gray'))

        ax.set_xlabel(names[0], fontsize=11, labelpad=8)
        ax.set_ylabel(names[1], fontsize=11, labelpad=8)
        ax.set_zlabel(names[2], fontsize=11, labelpad=8)

        if len(verts) > 0:
            margin = 0.3
            ax.set_xlim([0, max(verts[:, 0].max(), 0.1) + margin])
            ax.set_ylim([0, max(verts[:, 1].max(), 0.1) + margin])
            ax.set_zlim([0, max(verts[:, 2].max(), 0.1) + margin])

        ax.legend(fontsize=9, loc='upper left')
        title = f"Iterazione {step}"
        if step == self.total_steps - 1:
            title += " — OTTIMO"
        ax.set_title(title, fontsize=13, pad=10)

    def _plot_2d(self, ax, step):
        """Disegna la regione ammissibile 2D e il percorso del simplesso."""
        verts = self._find_vertices()
        names = self.solver.var_names
        A = self.solver.A_orig
        b = self.solver.b_orig

        if len(verts) > 0:
            max_x = max(verts[:, 0].max() * 1.3, 1)
            max_y = max(verts[:, 1].max() * 1.3, 1)
        else:
            max_x, max_y = 10, 10

        x_range = np.linspace(0, max_x * 1.2, 400)

        # Disegna i vincoli come linee
        colors_line = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2',
                       '#C62828', '#00838F', '#4E342E', '#1565C0']
        for i in range(len(b)):
            a1, a2 = A[i, 0], A[i, 1]
            bi = b[i]
            color = colors_line[i % len(colors_line)]
            s_sym = '≤' if self.solver.signs_orig[i] == '<=' else ('≥' if self.solver.signs_orig[i] == '>=' else '=')
            if abs(a2) > 1e-10:
                y_line = (bi - a1 * x_range) / a2
                mask = (y_line >= -0.5) & (y_line <= max_y * 1.3)
                lbl = f"{self._frac(a1)}x₁+{self._frac(a2)}x₂ {s_sym} {self._frac(bi)}"
                ax.plot(x_range[mask], y_line[mask], color=color,
                        linewidth=1.5, alpha=0.7, label=lbl)
            elif abs(a1) > 1e-10:
                x_val = bi / a1
                ax.axvline(x=x_val, color=color, linewidth=1.5, alpha=0.7,
                           label=f"x₁ {s_sym} {self._frac(x_val)}")

        # Regione ammissibile
        if len(verts) >= 3:
            center = np.mean(verts, axis=0)
            angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
            order = np.argsort(angles)
            sorted_v = verts[order]
            from matplotlib.patches import Polygon
            poly = Polygon(sorted_v, alpha=0.18, facecolor='cyan',
                           edgecolor='navy', linewidth=1.5)
            ax.add_patch(poly)

        # Percorso simplesso
        points = [self.solver.history[s]['decision_vars'][:2]
                  for s in range(step + 1)]
        pts = np.array(points)

        if len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=2.5, zorder=8, alpha=0.8)

        for i, pt in enumerate(pts):
            if i == 0:
                color, marker, size = COL['vertex_start'], '^', 120
                label = "Start"
            elif i == step and step == self.total_steps - 1:
                color, marker, size = COL['vertex_optimal'], '*', 200
                label = "Ottimo"
            elif i == step:
                color, marker, size = COL['vertex_current'], 'D', 100
                label = "Corrente"
            else:
                color, marker, size = '#FF8A80', 'o', 60
                label = None

            ax.scatter(*pt, c=color, s=size, marker=marker, zorder=10,
                       edgecolors='black', linewidths=1,
                       label=label if label else None)

            z_val = self.solver.history[i]['z_value']
            if self.is_min:
                z_val = -z_val
            ax.annotate(f"It.{i}\nz={self._frac(z_val)}", pt,
                        textcoords="offset points", xytext=(12, 12),
                        fontsize=9, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  alpha=0.85, edgecolor='gray'))

        ax.set_xlabel(names[0], fontsize=12)
        ax.set_ylabel(names[1], fontsize=12)
        ax.set_xlim([-0.3, max_x * 1.15])
        ax.set_ylim([-0.3, max_y * 1.15])
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')
        title = f"Iterazione {step}"
        if step == self.total_steps - 1:
            title += " — OTTIMO"
        ax.set_title(title, fontsize=13)

    # ═══════════════════════════════════════════════════
    #  Controlli navigazione
    # ═══════════════════════════════════════════════════

    def _step_forward(self):
        if self.solver and self.current_step < self.total_steps - 1:
            self._display_step(self.current_step + 1)

    def _step_backward(self):
        if self.solver and self.current_step > 0:
            self._display_step(self.current_step - 1)

    def _toggle_animation(self):
        if self.anim_id is not None:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        if not self.solver:
            return
        self.play_btn.config(text="Pausa", bg='#FFE0B2', fg=COL['warning'])
        self._animate_tick()

    def _stop_animation(self):
        if self.anim_id is not None:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None
        self.play_btn.config(text="Play", bg=COL['succ_lt'], fg=COL['success'])

    def _animate_tick(self):
        """Avanza di un passo nell'animazione."""
        if self.current_step < self.total_steps - 1:
            self._step_forward()
            self.anim_id = self.root.after(self.speed_ms.get(), self._animate_tick)
        else:
            self._stop_animation()

    def _update_speed_label(self, *args):
        try:
            ms = self.speed_ms.get()
            self.speed_val_label.config(text=f"{ms/1000:.1f}s")
        except tk.TclError:
            pass

    def _reset(self):
        """Reset completo."""
        self._stop_animation()
        self.solver = None
        self.current_step = 0
        self.total_steps = 0

        self.step_label.config(text="Premi RISOLVI per iniziare", fg=COL['text'])

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Inserisci un problema e premi RISOLVI",
                ha='center', va='center', fontsize=14, color=COL['text2'],
                transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas_mpl.draw()

        t = self.tableau_text
        t.config(state=tk.NORMAL)
        t.delete('1.0', tk.END)
        t.insert(tk.END, "\n   Il tableau apparirà qui dopo aver premuto RISOLVI\n",
                 'info')
        t.config(state=tk.DISABLED)


# ═══════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app = SimplexGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
