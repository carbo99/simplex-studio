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
    import tempfile
    os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl')

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from fractions import Fraction
import platform

import matplotlib
if os.environ.get('MPLBACKEND', '').lower() != 'agg':
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

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Importa il solver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simplex import SimplexSolver


def asset_path(filename):
    """Find bundled or source artwork, independently of the working directory."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    bases = [getattr(sys, '_MEIPASS', module_dir), module_dir]
    if getattr(sys, 'frozen', False):
        bases.append(os.path.dirname(sys.executable))
    for base in dict.fromkeys(bases):
        for relative in (('docs', 'img'), ('img',)):
            candidate = os.path.join(base, *relative, filename)
            if os.path.isfile(candidate):
                return candidate
    return os.path.join(module_dir, 'docs', 'img', filename)


def set_window_icon(root):
    """Retain the Tk image and use the native multi-size ICO on Windows."""
    png = asset_path('icon.png')
    if os.path.isfile(png):
        try:
            if HAS_PIL:
                with Image.open(png) as source:
                    photo = ImageTk.PhotoImage(source.resize((64, 64), Image.Resampling.LANCZOS))
            else:
                photo = tk.PhotoImage(master=root, file=png)
            root.iconphoto(True, photo)
            root._simplex_icon = photo
        except (OSError, ValueError, tk.TclError):
            pass
    ico = asset_path('icon.ico')
    if sys.platform == 'win32' and os.path.isfile(ico):
        try:
            root.iconbitmap(default=ico)
        except tk.TclError:
            pass

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

        set_window_icon(self.root)

        # Dimensione finestra
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1560, sw - 40), min(950, sh - 40)
        root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        root.minsize(1100, 640)

        # ── Stato ──
        self.n_vars = tk.IntVar(value=3)
        self.n_constr = tk.IntVar(value=3)
        self.obj_type = tk.StringVar(value='max')
        self.pivot_rule = tk.StringVar(value='Dantzig')
        self.speed_ms = tk.IntVar(value=1200)
        self.view_mode = tk.StringVar(value='original')  # 'original' o 'non_basic'

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
        root.bind('<Right>', lambda e: self._shortcut(e, self._step_forward))
        root.bind('<Left>', lambda e: self._shortcut(e, self._step_backward))
        root.bind('<space>', lambda e: self._shortcut(e, self._toggle_animation))
        root.bind('<Escape>', lambda e: self._stop_animation())

    # ═══════════════════════════════════════════════════
    #  Costruzione UI
    # ═══════════════════════════════════════════════════

    def _shortcut(self, event, action):
        if not isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Text)):
            action()

    def _build_ui(self):
        # Paned window principale (sinistra / destra)
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                     bg=COL['border'], sashwidth=4,
                                     sashrelief=tk.FLAT)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── Pannello sinistro (input) ──
        left = tk.Frame(self.paned, bg=COL['bg'], width=520)
        self.paned.add(left, minsize=480, width=520)
        self._build_left(left)

        # ── Pannello destro (visualizzazione) ──
        right = tk.Frame(self.paned, bg=COL['bg'])
        self.paned.add(right, minsize=660)
        self._build_right(right)

    # ─── Pannello sinistro ────────────────────────────

    def _build_left(self, parent):
        # 1. Contenitore inferiore FISSO (Pulsanti Azione + Logo + Copyright)
        # Viene impacchettato per PRIMO con side=tk.BOTTOM per garantire
        # che rimanga sempre visibile e fisso a qualsiasi risoluzione.
        bottom_fixed = tk.Frame(parent, bg=COL['bg'], bd=1, relief=tk.GROOVE)
        bottom_fixed.pack(fill=tk.X, side=tk.BOTTOM, padx=6, pady=(0, 6))

        # ── Pulsanti azione (fissi in basso, affiancati su un'unica riga per ottimizzare lo spazio) ──
        btn_frame = tk.Frame(bottom_fixed, bg=COL['bg'])
        btn_frame.pack(fill=tk.X, padx=8, pady=(6, 4))

        # Pulsante RISOLVI (principale, largo e ben visibile)
        self.solve_btn = tk.Label(btn_frame, text="RISOLVI",
                                  font=(FONT_UI_BOLD[0], FONT_UI_BOLD[1] + 1, 'bold'),
                                  bg='#D32F2F', fg='#FFFFFF',
                                  cursor='hand2', pady=7, relief=tk.RAISED, bd=1)
        self.solve_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.solve_btn.bind('<Button-1>', lambda e: self._solve())
        self.solve_btn.bind('<Enter>', lambda e: self.solve_btn.config(bg='#B71C1C'))
        self.solve_btn.bind('<Leave>', lambda e: self.solve_btn.config(bg='#D32F2F'))

        # Pulsante Reset (compatto affiancato a destra)
        self.reset_btn = tk.Label(btn_frame, text="Reset", font=FONT_UI,
                                  bg='#ECEFF1', fg=COL['text2'],
                                  cursor='hand2', padx=14, pady=7, relief=tk.GROOVE, bd=1)
        self.reset_btn.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        self.reset_btn.bind('<Button-1>', lambda e: self._reset())
        self.reset_btn.bind('<Enter>', lambda e: self.reset_btn.config(bg='#CFD8DC'))
        self.reset_btn.bind('<Leave>', lambda e: self.reset_btn.config(bg='#ECEFF1'))

        # Separatore visivo sottile
        sep = tk.Frame(bottom_fixed, height=1, bg=COL['border'])
        sep.pack(fill=tk.X, padx=10, pady=(3, 3))

        # ── Logo SimplexStudio & Copyright ben visibili ──
        brand_frame = tk.Frame(bottom_fixed, bg=COL['bg'])
        brand_frame.pack(fill=tk.X, padx=6, pady=(0, 4))

        logo_path = asset_path('logo.png')
        if HAS_PIL and os.path.exists(logo_path):
            try:
                logo_img_raw = Image.open(logo_path)
                # Rimuove i margini trasparenti vuoti per sfruttare al 100% lo spazio
                bbox = logo_img_raw.getbbox()
                if bbox:
                    logo_img_raw = logo_img_raw.crop(bbox)
                lw, lh = logo_img_raw.size
                target_h = 135
                target_w = int(lw * (target_h / lh))
                logo_img_resized = logo_img_raw.resize((target_w, target_h), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img_resized)
                logo_label = tk.Label(brand_frame, image=self.logo_photo, bg=COL['bg'])
                logo_label.pack(anchor='center', pady=(2, 2))
            except Exception:
                pass

        FONT_COPY = ('Helvetica Neue', 11) if IS_MAC else ('Segoe UI', 10)
        tk.Label(brand_frame,
                 text="© 2026 Alessandro Carboni\ncarboni.alessandro@outlook.it",
                 font=FONT_COPY, bg=COL['bg'], fg=COL['text2'],
                 justify=tk.CENTER).pack(anchor='center', pady=(2, 4))

        # 2. Contenitore superiore: scrollabile (Configurazione, Obiettivo, Vincoli, Esempi)
        top_container = tk.Frame(parent, bg=COL['bg'])
        top_container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(top_container, bg=COL['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(top_container, orient=tk.VERTICAL, command=canvas.yview)
        self.left_inner = tk.Frame(canvas, bg=COL['bg'])

        self.left_inner.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas_win = canvas.create_window((0, 0), window=self.left_inner, anchor='nw')

        # Assicura che la larghezza dell'inner frame si adatti al canvas
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_win, width=event.width)
        canvas.bind('<Configure>', _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scroll con mousewheel solo all'interno del pannello sinistro
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")
        self.left_inner.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', _on_mousewheel))
        self.left_inner.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

        # ── Popola l'area scrollabile (f = self.left_inner) ──
        f = self.left_inner
        pad = {'padx': 8, 'pady': 3}

        # ── Titolo ──
        tk.Label(f, text="Configurazione", font=FONT_TITLE,
                 bg=COL['bg'], fg=COL['primary']).pack(anchor='w', padx=10, pady=(8, 6))

        # ── Configurazione ──
        cfg = tk.LabelFrame(f, text=" Problema ", font=FONT_SECTION,
                            bg=COL['card'], fg=COL['text'], bd=1,
                            relief=tk.GROOVE, padx=10, pady=6)
        cfg.pack(fill=tk.X, **pad)

        r = 0
        tk.Label(cfg, text="N° variabili:", font=FONT_UI, bg=COL['card']).grid(
            row=r, column=0, sticky='w', pady=2)
        vars_cb = ttk.Combobox(cfg, textvariable=self.n_vars, values=[2, 3, 4, 5, 6],
                               width=5, state='readonly', font=FONT_UI)
        vars_cb.grid(row=r, column=1, sticky='w', padx=(8, 0), pady=2)
        vars_cb.bind('<<ComboboxSelected>>', lambda e: self._rebuild_form())

        r += 1
        tk.Label(cfg, text="N° vincoli:", font=FONT_UI, bg=COL['card']).grid(
            row=r, column=0, sticky='w', pady=2)
        constr_cb = ttk.Combobox(cfg, textvariable=self.n_constr,
                                 values=list(range(1, 11)),
                                 width=5, state='readonly', font=FONT_UI)
        constr_cb.grid(row=r, column=1, sticky='w', padx=(8, 0), pady=2)
        constr_cb.bind('<<ComboboxSelected>>', lambda e: self._rebuild_form())

        r += 1
        tk.Label(cfg, text="Obiettivo:", font=FONT_UI, bg=COL['card']).grid(
            row=r, column=0, sticky='w', pady=2)
        obj_cb = ttk.Combobox(cfg, textvariable=self.obj_type,
                              values=['max', 'min'], width=5,
                              state='readonly', font=FONT_UI)
        obj_cb.grid(row=r, column=1, sticky='w', padx=(8, 0), pady=2)

        r += 1
        tk.Label(cfg, text="Regola pivot:", font=FONT_UI, bg=COL['card']).grid(
            row=r, column=0, sticky='w', pady=2)
        rule_cb = ttk.Combobox(cfg, textvariable=self.pivot_rule,
                               values=['Dantzig', 'Bland (anti-ciclo)'], width=15,
                               state='readonly', font=FONT_UI)
        rule_cb.grid(row=r, column=1, sticky='w', padx=(8, 0), pady=2)

        # ── Funzione obiettivo ──
        self.obj_frame = tk.LabelFrame(f, text=" Funzione obiettivo ",
                                        font=FONT_SECTION, bg=COL['card'],
                                        fg=COL['text'], bd=1, relief=tk.GROOVE,
                                        padx=6, pady=4)
        self.obj_frame.pack(fill=tk.X, **pad)

        # ── Vincoli ──
        self.constr_frame = tk.LabelFrame(f, text=" Vincoli ", font=FONT_SECTION,
                                           bg=COL['card'], fg=COL['text'], bd=1,
                                           relief=tk.GROOVE, padx=4, pady=4)
        self.constr_frame.pack(fill=tk.X, **pad)

        # ── Esempi ──
        ex_frame = tk.LabelFrame(f, text=" Esempi ", font=FONT_SECTION,
                                  bg=COL['card'], fg=COL['text'], bd=1,
                                  relief=tk.GROOVE, padx=10, pady=6)
        ex_frame.pack(fill=tk.X, **pad)

        tk.Button(ex_frame, text="Esempio in R²",
                  font=FONT_UI, bg='#E8F5E9', fg=COL['success'],
                  activebackground='#C8E6C9', relief=tk.FLAT, cursor='hand2',
                  command=lambda: self._load_example('2var')
                  ).pack(fill=tk.X, pady=2)
        tk.Button(ex_frame, text="Esempio in R³",
                  font=FONT_UI, bg='#E3F2FD', fg=COL['primary'],
                  activebackground='#BBDEFB', relief=tk.FLAT, cursor='hand2',
                  command=lambda: self._load_example('3var')
                  ).pack(fill=tk.X, pady=2)
        tk.Button(ex_frame, text="Esempio con Duale (R² ➔ R³)",
                  font=FONT_UI, bg='#F3E5F5', fg='#6A1B9A',
                  activebackground='#E1BEE7', relief=tk.FLAT, cursor='hand2',
                  command=lambda: self._load_example('dual_demo')
                  ).pack(fill=tk.X, pady=2)


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
        vpaned.add(bottom, minsize=220)

        # Controlli: organizzati su riga superiore (navigazione) e riga inferiore (parametri) per massima leggibilità
        ctrl = tk.Frame(bottom, bg=COL['card'], bd=1, relief=tk.GROOVE, padx=12, pady=6)
        ctrl.pack(fill=tk.X, padx=2, pady=(2, 4))

        # Riga 1 dei controlli: stato + pulsanti di navigazione
        ctrl_r1 = tk.Frame(ctrl, bg=COL['card'])
        ctrl_r1.pack(fill=tk.X, pady=(0, 4))

        self.step_label = tk.Label(ctrl_r1, text="Premi RISOLVI per iniziare",
                                    font=FONT_UI_BOLD, bg=COL['card'], fg=COL['text'])
        self.step_label.pack(side=tk.LEFT, padx=(0, 15))

        self.play_btn = tk.Button(ctrl_r1, text="Play", font=FONT_UI_BOLD,
                                   bg=COL['succ_lt'], fg=COL['success'],
                                   relief=tk.FLAT, cursor='hand2',
                                   command=self._toggle_animation)
        self.play_btn.pack(side=tk.RIGHT, padx=4)

        tk.Button(ctrl_r1, text="Avanti", font=FONT_UI, bg='#E0E0E0',
                  relief=tk.FLAT, cursor='hand2',
                  command=self._step_forward).pack(side=tk.RIGHT, padx=4)

        tk.Button(ctrl_r1, text="Indietro", font=FONT_UI, bg='#E0E0E0',
                  relief=tk.FLAT, cursor='hand2',
                  command=self._step_backward).pack(side=tk.RIGHT, padx=4)

        # Riga 2 dei controlli: velocità e selettore spazio assi
        ctrl_r2 = tk.Frame(ctrl, bg=COL['card'])
        ctrl_r2.pack(fill=tk.X, pady=(2, 0))

        tk.Label(ctrl_r2, text="Velocità animazione:", font=FONT_UI, bg=COL['card'], fg=COL['text2']).pack(
            side=tk.LEFT, padx=(0, 4))
        speed_scale = ttk.Scale(ctrl_r2, from_=200, to=3000, variable=self.speed_ms,
                                 orient=tk.HORIZONTAL, length=120)
        speed_scale.pack(side=tk.LEFT, padx=3)
        self.speed_val_label = tk.Label(ctrl_r2, text="1.2s", font=FONT_MONO_SM,
                                         bg=COL['card'], fg=COL['text2'], width=5)
        self.speed_val_label.pack(side=tk.LEFT, padx=(0, 20))
        self.speed_ms.trace_add('write', self._update_speed_label)

        # Selettore spazio di visualizzazione (feature extra)
        tk.Label(ctrl_r2, text="Spazio Assi:", font=FONT_UI_BOLD, bg=COL['card'], fg=COL['primary']).pack(
            side=tk.LEFT, padx=(0, 4))
        self.view_cb = ttk.Combobox(ctrl_r2, textvariable=self.view_mode,
                                    values=["Variabili Originali", "Nuovi Assi (Fuori Base)"],
                                    state='readonly', width=22, font=FONT_UI)
        self.view_cb.set("Variabili Originali")
        self.view_cb.pack(side=tk.LEFT, padx=2)
        self.view_cb.bind('<<ComboboxSelected>>', lambda e: self._on_view_mode_changed())

        # Pulsante Analisi Duale
        self.dual_btn = tk.Button(ctrl_r2, text="Analisi Duale", font=FONT_UI_BOLD,
                                  bg='#EDE7F6', fg='#512DA8',
                                  activebackground='#D1C4E9', relief=tk.FLAT, cursor='hand2',
                                  command=self._open_dual_window)
        self.dual_btn.pack(side=tk.RIGHT, padx=4)

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
        """Ricostruisce i campi di input preservando i valori già inseriti dall'utente."""
        n = self.n_vars.get()
        m = self.n_constr.get()

        # Salva i valori attuali della funzione obiettivo
        old_obj = [e.get().strip() for e in self.obj_entries]

        # Salva i valori attuali dei vincoli
        old_constrs = []
        for row in self.constr_rows:
            coeffs = [e.get().strip() for e in row['coeffs']]
            sign = row['sign'].get()
            rhs = row['rhs'].get().strip()
            old_constrs.append({'coeffs': coeffs, 'sign': sign, 'rhs': rhs})

        # ── Funzione obiettivo ──
        for w in self.obj_frame.winfo_children():
            w.destroy()
        self.obj_entries = []

        row = tk.Frame(self.obj_frame, bg=COL['card'])
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="z =", font=FONT_UI_BOLD, bg=COL['card']).pack(
            side=tk.LEFT, padx=(0, 5))

        # Larghezza entry adattiva al numero di variabili
        entry_w = 4 if n <= 2 else (3 if n <= 4 else 2)

        for i in range(n):
            if i > 0:
                tk.Label(row, text=" + ", font=FONT_UI, bg=COL['card']).pack(side=tk.LEFT, padx=0)
            e = tk.Entry(row, width=entry_w, font=FONT_MONO, justify=tk.CENTER,
                         relief=tk.SOLID, bd=1)
            e.pack(side=tk.LEFT, padx=1)
            # Ripristina vecchio valore o metti default '0'
            val = old_obj[i] if i < len(old_obj) and old_obj[i] != "" else "0"
            e.insert(0, val)
            self.obj_entries.append(e)
            tk.Label(row, text=f"x{i+1}", font=FONT_UI, bg=COL['card'],
                     fg=COL['primary']).pack(side=tk.LEFT, padx=(0, 2))

        # ── Vincoli ──
        for w in self.constr_frame.winfo_children():
            w.destroy()
        self.constr_rows = []

        for i in range(m):
            saved_data = old_constrs[i] if i < len(old_constrs) else None
            self._add_constraint_row(i, n, initial_data=saved_data)

    def _add_constraint_row(self, idx, n, initial_data=None):
        """Aggiunge una riga vincolo al form, ripristinando i valori se forniti."""
        row_frame = tk.Frame(self.constr_frame, bg=COL['card'])
        row_frame.pack(fill=tk.X, pady=2)

        tk.Label(row_frame, text=f"V{idx+1}:", font=FONT_UI,
                 bg=COL['card'], fg=COL['text2'], width=3).pack(side=tk.LEFT)

        # Larghezza entry adattiva per garantire che da 2 fino a 6 variabili entrino perfettamente senza tagli
        entry_w = 4 if n <= 2 else (3 if n <= 4 else 2)
        font_c = FONT_MONO if n <= 4 else FONT_MONO_SM
        font_u = FONT_UI if n <= 4 else FONT_MONO_SM

        saved_coeffs = initial_data['coeffs'] if initial_data else []
        saved_sign = initial_data['sign'] if initial_data else '≤'
        saved_rhs = initial_data['rhs'] if initial_data else '0'

        coeff_entries = []
        for j in range(n):
            if j > 0:
                tk.Label(row_frame, text="+", font=font_u,
                         bg=COL['card'], fg=COL['text2']).pack(side=tk.LEFT, padx=0)
            e = tk.Entry(row_frame, width=entry_w, font=font_c, justify=tk.CENTER,
                         relief=tk.SOLID, bd=1)
            e.pack(side=tk.LEFT, padx=0)
            val = saved_coeffs[j] if j < len(saved_coeffs) and saved_coeffs[j] != "" else "0"
            e.insert(0, val)
            coeff_entries.append(e)
            tk.Label(row_frame, text=f"x{j+1}", font=font_u, bg=COL['card'],
                     fg=COL['primary']).pack(side=tk.LEFT, padx=(0, 1))

        # Segno del vincolo
        sign_var = tk.StringVar(value=saved_sign)
        sign_cb = ttk.Combobox(row_frame, textvariable=sign_var,
                                values=['≤', '≥', '='], width=2,
                                state='readonly', font=font_c)
        sign_cb.pack(side=tk.LEFT, padx=1)

        # Termine noto
        rhs_e = tk.Entry(row_frame, width=3 if n >= 5 else 4, font=font_c, justify=tk.CENTER,
                         relief=tk.SOLID, bd=1)
        rhs_e.pack(side=tk.LEFT, padx=1)
        rhs_e.insert(0, saved_rhs if saved_rhs != "" else "0")

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
        elif name == 'dual_demo':
            # Esempio ideale per la Dualità: 2 variabili (Primale 2D) e 3 vincoli (Duale 3D)
            # max z = 3x1 + 2x2
            # s.t. 2x1 + x2 <= 10, x1 + 2x2 <= 8, -x1 + x2 <= 2, x1,x2 >= 0
            self.n_vars.set(2)
            self.n_constr.set(3)
            self.obj_type.set('max')
            self._rebuild_form()
            c = [3, 2]
            A = [
                [2, 1],
                [1, 2],
                [-1, 1]
            ]
            b = [10, 8, 2]
            signs = ['≤', '≤', '≤']
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
        """Converte testo in float. Supporta frazioni come 1/2, 1.5/2, -3/4 o decimali 0.5."""
        from exact_engine import number
        value = float(number(text.strip() or '0'))
        if not np.isfinite(value):
            raise ValueError('Il valore deve essere finito.')
        return value

    def _parse_inputs(self):
        """Legge i valori dai campi e restituisce c, A, b, signs, is_min."""
        from exact_engine import number
        parse = lambda text: number(text.strip() or "0")
        n = self.n_vars.get()

        # Obiettivo
        c = []
        for e in self.obj_entries:
            c.append(parse(e.get()))

        # Se min, converti in max negando c
        is_min = self.obj_type.get() == 'min'
        if is_min:
            c = [-ci for ci in c]

        # Vincoli originali con i loro segni
        A = []
        b = []
        signs = []
        for row in self.constr_rows:
            coeffs = [parse(e.get()) for e in row['coeffs']]
            rhs = parse(row['rhs'].get())
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
        # Regola di pivot selezionata
        p_rule = 'dantzig' if 'dantzig' in self.pivot_rule.get().lower() else 'bland'
        try:
            candidate = SimplexSolver(c, A, b, signs=signs, var_names=var_names, pivot_rule=p_rule, is_min=is_min)
            result = candidate.solve(verbose=False, plot3d=False)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la risoluzione:\n{e}")
            return

        self.solver = candidate
        self.is_min = is_min
        self.total_steps = len(self.solver.history)
        self.current_step = 0

        if result is None:
            last_desc = self.solver.history[-1]['description'] if self.solver.history else ""
            if "Inammissibile" in last_desc:
                messagebox.showwarning("Inammissibile",
                    "L'insieme dei vincoli non ammette soluzioni ammissibili (regione ammissibile vuota).")
            elif "Limite iterazioni" in last_desc:
                messagebox.showwarning("Limite Iterazioni Raggiunto",
                    "Raggiunto il numero massimo di iterazioni consentite (possibile ciclo con regola Dantzig).\n\nProva a selezionare la 'Regola pivot: Bland (anti-ciclo)'.")
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
        status = getattr(self.solver, 'status', 'optimal')
        if step == self.total_steps - 1:
            if status == 'infeasible' or "Inammissibile" in state['description']:
                self.step_label.config(
                    text="Inammissibile: l'insieme dei vincoli non ammette soluzioni",
                    fg=COL['danger'])
            elif status == 'unbounded' or "illimitato" in state['description'].lower():
                self.step_label.config(
                    text=f"{prefix}Passo {step}/{self.total_steps-1} (ILLIMITATO)",
                    fg=COL['danger'])
            elif status == 'max_iterations' or "Limite iterazioni" in state['description']:
                self.step_label.config(
                    text=f"{prefix}Passo {step}/{self.total_steps-1} (LIMITE ITERAZIONI)",
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
        return str(val) if isinstance(val, Fraction) else str(Fraction(float(val)).limit_denominator(10**9))

    def _render_tableau(self, state, step):
        """Genera il tableau nel widget Text con colori."""
        t = self.tableau_text
        t.config(state=tk.NORMAL)
        t.delete('1.0', tk.END)

        tableau = np.array(state.get('exact_tableau', state['tableau']), dtype=object)
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
        dec = state.get('exact_solution', state['decision_vars'])[:self.solver.n]
        z_val = state.get('exact_z', state['z_value'])
        if self.is_min and state.get('phase', 2) == 2:
            z_val = -z_val
        n_dec = self.solver.n
        sol_parts = [f"{self.solver.var_names[i]}={self._frac(dec[i])}"
                     for i in range(n_dec)]

        t.insert(tk.END, f"  Soluzione {'ausiliaria' if state.get('phase') == 1 else 'di base ammissibile'}: ({', '.join(sol_parts)})\n", 'info')

        lbl_display = obj_lbl.replace('-', '')
        z_display = self._frac(z_val)
        status = getattr(self.solver, 'status', 'optimal')
        is_opt = (step == self.total_steps - 1 and status == 'optimal')
        tag = 'optimal' if is_opt else 'info'
        t.insert(tk.END, f"  {lbl_display} = {z_display}", tag)

        if step == self.total_steps - 1:
            if status == 'optimal':
                t.insert(tk.END, "  [OTTIMO]", 'optimal')
            elif status == 'unbounded':
                t.insert(tk.END, "  [PROBLEMA ILLIMITATO]", 'leaving')
            elif status == 'infeasible':
                t.insert(tk.END, "  [PROBLEMA INAMMISSIBILE]", 'leaving')
            elif status == 'max_iterations':
                t.insert(tk.END, "  [LIMITE ITERAZIONI RAGGIUNTO]", 'leaving')

        t.insert(tk.END, "\n")
        basis_names = [names[b] for b in basis if b < len(names)]
        t.insert(tk.END, f"  Base: {{{', '.join(basis_names)}}}\n", 'info')

        # Se siamo all'ottimo, mostra anche i Prezzi Ombra (Variabili Duali Ottime y*)
        if is_opt and self.solver:
            y_opt = self.solver.get_dual_solution()
            if y_opt is not None:
                t.insert(tk.END, "\n  Variabili Duali Ottime y* (Prezzi Ombra):\n", 'optimal')
                for i in range(len(y_opt)):
                    val_y = self._frac(y_opt[i])
                    sign_desc = self.solver.signs_orig[i]
                    b_val = self._frac(self.solver.b_orig[i])
                    t.insert(tk.END, f"    y{i+1} = {val_y:<8} [associata a Vincolo V{i+1}: {sign_desc} {b_val}]\n", 'info')

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

    def _on_view_mode_changed(self):
        """Callback al cambio della modalità di visualizzazione degli assi."""
        if self.solver:
            self._render_plot(self.current_step)

    def _render_plot(self, step):
        """Aggiorna il grafico matplotlib."""
        self.fig.clear()
        n = self.solver.n
        mode = self.view_cb.get() if hasattr(self, 'view_cb') else "Variabili Originali"

        if "Nuovi Assi" in mode:
            self._render_plot_non_basic(step)
            return

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

    def _get_full_variables_for_point(self, x):
        """Calcola il valore di tutte le variabili (decisionali + slack) per un vertice x."""
        full = list(x)
        for (r, coeff) in self.solver.slack_cols:
            val = (self.solver.b_norm[r] - np.dot(self.solver.A_norm[r], x)) / coeff
            full.append(val)
        full.extend([0.0] * len(self.solver.art_names))
        return np.array(full)

    def _render_plot_non_basic(self, step):
        """Visualizza il poliedro nello spazio delle variabili fuori base dell'iterazione corrente."""
        state = self.solver.history[step]
        all_names = state['all_var_names']
        basis = [int(b) for b in state['basis']]
        nb_indices = [j for j in range(len(all_names)) if j not in basis]
        nb_names = [all_names[j] for j in nb_indices]

        dim = len(nb_indices)
        verts = self._find_vertices()
        extent = max([5.0] + [float(v.max()) * 1.3 for v in verts] +
                     [float(h['decision_vars'].max()) * 1.3 for h in self.solver.history])
        verts = self.solver.display_vertices([(0, extent)] * self.solver.n)

        if len(verts) == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Nessun vertice ammissibile", ha='center', va='center',
                    fontsize=14, color=COL['text2'], transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas_mpl.draw()
            return

        # Trasforma tutti i vertici nelle coordinate fuori base
        transformed_verts = []
        for v in verts:
            full = self._get_full_variables_for_point(v)
            if nb_indices and max(nb_indices) < len(full):
                transformed_verts.append(full[nb_indices])
            else:
                transformed_verts.append(np.zeros(dim))
        transformed_verts = np.array(transformed_verts)

        # Vertice corrente: per definizione di BFS, le variabili fuori base valgono 0!
        current_pt = np.zeros(dim)

        if dim == 3:
            ax = self.fig.add_subplot(111, projection='3d')
            if len(transformed_verts) >= 4 and HAS_SCIPY:
                try:
                    hull = ConvexHull(transformed_verts)
                    faces = [[transformed_verts[i] for i in simplex] for simplex in hull.simplices]
                    poly = Poly3DCollection(faces, alpha=0.18, facecolor='#FF9800',
                                             edgecolor='#E65100', linewidth=1.2)
                    ax.add_collection3d(poly)
                except Exception:
                    pass

            ax.scatter(transformed_verts[:, 0], transformed_verts[:, 1], transformed_verts[:, 2],
                       c='navy', s=25, alpha=0.5, zorder=5)

            # Evidenzia il vertice corrente (nell'origine degli assi fuori base)
            color = COL['vertex_optimal'] if step == self.total_steps - 1 and self.solver.status == 'optimal' else COL['vertex_current']
            marker = '*' if step == self.total_steps - 1 and self.solver.status == 'optimal' else 'D'
            size = 200 if step == self.total_steps - 1 and self.solver.status == 'optimal' else 130
            ax.scatter([0], [0], [0], c=color, s=size, marker=marker,
                       edgecolors='black', linewidths=1.5, zorder=15,
                       label=f"Vertice It.{step} (Origine)")

            ax.set_xlabel(f"Asse {nb_names[0]}", fontsize=11, labelpad=8, fontweight='bold', color=COL['primary'])
            ax.set_ylabel(f"Asse {nb_names[1]}", fontsize=11, labelpad=8, fontweight='bold', color=COL['primary'])
            ax.set_zlabel(f"Asse {nb_names[2]}", fontsize=11, labelpad=8, fontweight='bold', color=COL['primary'])

            max_val = max(transformed_verts.max() * 1.1, 1.0)
            ax.set_xlim([-0.2, max_val])
            ax.set_ylim([-0.2, max_val])
            ax.set_zlim([-0.2, max_val])

            ax.legend(fontsize=9, loc='upper left')
            ax.set_title(f"Spazio Nuovi Assi Fuori Base ({', '.join(nb_names)}) — Iterazione {step}",
                         fontsize=12, pad=10, fontweight='bold')

        elif dim == 2:
            ax = self.fig.add_subplot(111)
            if len(transformed_verts) >= 3:
                try:
                    from matplotlib.patches import Polygon
                    center = np.mean(transformed_verts, axis=0)
                    angles = np.arctan2(transformed_verts[:, 1] - center[1], transformed_verts[:, 0] - center[0])
                    sorted_v = transformed_verts[np.argsort(angles)]
                    poly = Polygon(sorted_v, alpha=0.22, facecolor='#FF9800',
                                   edgecolor='#E65100', linewidth=1.5)
                    ax.add_patch(poly)
                except Exception:
                    pass

            ax.scatter(transformed_verts[:, 0], transformed_verts[:, 1],
                       c='navy', s=30, alpha=0.5, zorder=5)

            # Evidenzia vertice corrente (Origine)
            color = COL['vertex_optimal'] if step == self.total_steps - 1 and self.solver.status == 'optimal' else COL['vertex_current']
            marker = '*' if step == self.total_steps - 1 and self.solver.status == 'optimal' else 'D'
            size = 200 if step == self.total_steps - 1 and self.solver.status == 'optimal' else 130
            ax.scatter([0], [0], c=color, s=size, marker=marker,
                       edgecolors='black', linewidths=1.5, zorder=15,
                       label=f"Vertice It.{step} (Origine)")

            ax.set_xlabel(f"Asse {nb_names[0]}", fontsize=12, fontweight='bold', color=COL['primary'])
            ax.set_ylabel(f"Asse {nb_names[1]}", fontsize=12, fontweight='bold', color=COL['primary'])

            max_x = max(transformed_verts[:, 0].max() * 1.15, 1.0)
            max_y = max(transformed_verts[:, 1].max() * 1.15, 1.0)
            ax.set_xlim([-0.3, max_x])
            ax.set_ylim([-0.3, max_y])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='upper right')
            ax.set_title(f"Spazio Nuovi Assi Fuori Base ({', '.join(nb_names)}) — Iterazione {step}",
                         fontsize=12, fontweight='bold')
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Spazio delle variabili fuori base a {dim} dimensioni\n(Assi: {', '.join(nb_names)})\nVisualizzazione grafica possibile per 2 o 3 assi.",
                    ha='center', va='center', fontsize=13, color=COL['text2'], transform=ax.transAxes)
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

        from geometry import draw_region
        extent = max(float(verts.max()) * 1.3, 5) if len(verts) else 10
        extent = max(extent, max(float(h['decision_vars'].max()) for h in self.solver.history) * 1.3)
        draw_region(ax, self.solver.display_vertices([(0, extent)] * 3), .12)

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
                elif i == step and step == self.total_steps - 1 and self.solver.status == 'optimal':
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

                h = self.solver.history[i]
                z_val = h.get('exact_z', h['z_value'])
                objective_label = 'w' if h['phase'] == 1 else 'z'
                if self.is_min and h['phase'] == 2:
                    z_val = -z_val
                ax.text(pt[0] + 0.08, pt[1] + 0.08, pt[2] + 0.08,
                        f"It.{i}\n{objective_label}={self._frac(z_val)}",
                        fontsize=8, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  alpha=0.85, edgecolor='gray'))

        ax.set_xlabel(names[0], fontsize=11, labelpad=8)
        ax.set_ylabel(names[1], fontsize=11, labelpad=8)
        ax.set_zlabel(names[2], fontsize=11, labelpad=8)

        ax.set_xlim(0, extent)
        ax.set_ylim(0, extent)
        ax.set_zlim(0, extent)

        ax.legend(fontsize=9, loc='upper left')
        status = getattr(self.solver, 'status', 'optimal')
        title = f"Iterazione {step}"
        if step == self.total_steps - 1:
            if status == 'optimal':
                title += " — OTTIMO"
            elif status == 'unbounded':
                title += " — ILLIMITATO"
            elif status == 'infeasible':
                title += " — INAMMISSIBILE"
            elif status == 'max_iterations':
                title += " — LIMITE ITERAZIONI"
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
        max_x = max(max_x, max(float(h['decision_vars'][0]) for h in self.solver.history) * 1.3)
        max_y = max(max_y, max(float(h['decision_vars'][1]) for h in self.solver.history) * 1.3)

        x_range = np.linspace(0, max_x * 1.2, 400)

        # Disegna i vincoli come linee
        colors_line = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2',
                       '#C62828', '#00838F', '#4E342E', '#1565C0']
        for i in range(len(b)):
            a1, a2 = A[i, 0], A[i, 1]
            bi = b[i]
            color = colors_line[i % len(colors_line)]
            s_sym = '≤' if self.solver.signs_orig[i] == '<=' else ('≥' if self.solver.signs_orig[i] == '>=' else '=')
            if a2 != 0:
                y_line = (bi - a1 * x_range) / a2
                mask = (y_line >= -0.5) & (y_line <= max_y * 1.3)
                lbl = f"{self._frac(a1)}x₁+{self._frac(a2)}x₂ {s_sym} {self._frac(bi)}"
                ax.plot(x_range[mask], y_line[mask], color=color,
                        linewidth=1.5, alpha=0.7, label=lbl)
            elif a1 != 0:
                x_val = bi / a1
                if a1 < 0:
                    s_sym = {'≤': '≥', '≥': '≤', '=': '='}[s_sym]
                ax.axvline(x=x_val, color=color, linewidth=1.5, alpha=0.7,
                           label=f"x₁ {s_sym} {self._frac(x_val)}")

        # Clip to viewport: a hull of LP vertices alone hides unbounded rays.
        from geometry import draw_region
        draw_region(ax, self.solver.display_vertices([(0, max_x * 1.15), (0, max_y * 1.15)]), .18)

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
            elif i == step and step == self.total_steps - 1 and self.solver.status == 'optimal':
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

            h = self.solver.history[i]
            z_val = h.get('exact_z', h['z_value'])
            objective_label = 'w' if h['phase'] == 1 else 'z'
            if self.is_min and h['phase'] == 2:
                z_val = -z_val
            ax.annotate(f"It.{i}\n{objective_label}={self._frac(z_val)}", pt,
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
        status = getattr(self.solver, 'status', 'optimal')
        title = f"Iterazione {step}"
        if step == self.total_steps - 1:
            if status == 'optimal':
                title += " — OTTIMO"
            elif status == 'unbounded':
                title += " — ILLIMITATO"
            elif status == 'infeasible':
                title += " — INAMMISSIBILE"
            elif status == 'max_iterations':
                title += " — LIMITE ITERAZIONI"
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
        """Reset completo: ferma animazioni, azzera i grafici, il tableau e tutti i campi di input."""
        self._stop_animation()
        self.solver = None
        self.current_step = 0
        self.total_steps = 0
        self.pivot_rule.set('Dantzig')

        # Ripristina tutti i coefficienti della funzione obiettivo a 0
        for e in self.obj_entries:
            e.delete(0, tk.END)
            e.insert(0, "0")

        # Ripristina tutti i vincoli a 0 ≤ 0
        for row in self.constr_rows:
            for e in row['coeffs']:
                e.delete(0, tk.END)
                e.insert(0, "0")
            row['sign'].set('≤')
            row['rhs'].delete(0, tk.END)
            row['rhs'].insert(0, "0")

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

    # ═══════════════════════════════════════════════════
    #  Finestra Analisi Duale
    # ═══════════════════════════════════════════════════

    def _open_dual_window(self):
        """Apre una finestra completa per lo studio della teoria della Dualità."""
        if not self.solver:
            messagebox.showinfo("Analisi Duale", "Inserisci e risolvi prima un problema lineare premendo RISOLVI.")
            return

        if getattr(self.solver, 'status', 'optimal') != 'optimal':
            status = getattr(self.solver, 'status', 'unbounded')
            msg = "il problema risulta inammissibile (nessuna soluzione ammissibile)." if status == 'infeasible' else "il problema risulta illimitato (funzione obiettivo non limitata superiormente)."
            messagebox.showwarning("Analisi Duale non disponibile",
                f"L'Analisi e la Geometria della Dualità richiedono una soluzione ottima finita.\n\nAttualmente {msg}")
            return

        dual_win = tk.Toplevel(self.root)
        dual_win.title("SimplexStudio – Analisi e Teoria della Dualità")
        dual_win.configure(bg=COL['bg'])
        dual_win.geometry("980x740")
        dual_win.minsize(850, 600)

        # Contenitore a schede (Notebook)
        nb = ttk.Notebook(dual_win)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Formulazione e Teoremi (Duale formale, Dualità forte, Scarti complementari)
        tab_theory = tk.Frame(nb, bg=COL['card'], padx=14, pady=12)
        nb.add(tab_theory, text=" Formulazione & Scarti Complementari ")

        # Tab 2: Grafico della Regione Duale (se le variabili duali m <= 3)
        tab_geom = tk.Frame(nb, bg=COL['card'], padx=10, pady=10)
        nb.add(tab_geom, text=" Geometria Duale (Poliedro) ")

        # ── Popola Tab 1 ──
        self._populate_dual_theory_tab(tab_theory)

        # ── Popola Tab 2 ──
        self._populate_dual_geom_tab(tab_geom)

    def _populate_dual_theory_tab(self, parent):
        """Costruisce il contenuto teorico, algebrico e di complementarità."""
        dual_info = self.solver.get_dual_formulation(is_primal_min=self.is_min)
        y_opt = self.solver.get_dual_solution()

        # Canvas con scrollbar per garantire leggibilità
        canvas = tk.Canvas(parent, bg=COL['card'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COL['card'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        f = scrollable_frame

        # ── Sezione 1: Teorema della Dualità Forte ──
        sec1 = tk.LabelFrame(f, text=" Teorema della Dualità Forte ", font=FONT_SECTION,
                             bg='#F3E5F5', fg='#4A148C', bd=1, relief=tk.GROOVE, padx=12, pady=10)
        sec1.pack(fill=tk.X, pady=(0, 10))

        primal_obj_name = "min" if self.is_min else "max"
        dec_vars, z_opt = self.solver.get_decision_variables()
        if self.is_min:
            z_opt = -z_opt

        # Valore ottimo duale: b^T y
        w_opt = sum(dual_info['obj_coeffs'][i] * y_opt[i] for i in range(len(y_opt))) if y_opt is not None else 0.0

        diff_dual = abs(z_opt - w_opt) if y_opt is not None else float('inf')
        if diff_dual < 1e-4:
            duality_verif = "✔ z* = w* (Scarto di dualità nullo: l'ottimo primale coincide esattamente con l'ottimo duale)"
            verif_fg = '#2E7D32'  # Verde successo
        else:
            duality_verif = f"✘ Scarto di dualità non nullo: |z* - w*| = {diff_dual:.4f}"
            verif_fg = '#C62828'  # Rosso errore

        lbl_text = (
            f"• Valore Ottimo Primale:  z* = {self._frac(z_opt)}\n"
            f"• Valore Ottimo Duale:    w* = {self._frac(w_opt)}\n"
            f"{duality_verif}"
        )
        tk.Label(sec1, text=lbl_text, font=FONT_MONO, bg='#F3E5F5', fg=verif_fg, justify=tk.LEFT).pack(anchor='w')

        # ── Sezione 2: Formulazione Algebrica del Problema Duale ──
        sec2 = tk.LabelFrame(f, text=" Problema Duale Associato ", font=FONT_SECTION,
                             bg=COL['card'], fg=COL['primary'], bd=1, relief=tk.GROOVE, padx=12, pady=10)
        sec2.pack(fill=tk.X, pady=(0, 10))

        # Funzione obiettivo duale
        obj_parts = []
        for i, bi in enumerate(dual_info['obj_coeffs']):
            bi_str = self._frac(bi)
            obj_parts.append(f"{bi_str}·y{i+1}")
        dual_obj_str = f"{dual_info['obj_type']} w = " + " + ".join(obj_parts)
        tk.Label(sec2, text=dual_obj_str, font=FONT_UI_BOLD, bg=COL['card'], fg=COL['text']).pack(anchor='w', pady=(0, 6))

        tk.Label(sec2, text="soggetto ai vincoli duali:", font=FONT_UI, bg=COL['card'], fg=COL['text2']).pack(anchor='w')

        # Vincoli duali: A^T y
        for j in range(len(dual_info['b'])):
            row_parts = []
            for i in range(len(y_opt)):
                aij = dual_info['A'][j, i]
                row_parts.append(f"{self._frac(aij)}·y{i+1}")
            sign_j = dual_info['constr_signs'][j]
            bj_val = self._frac(dual_info['b'][j])
            line = f"  V{j+1}_dual:  " + " + ".join(row_parts) + f"  {sign_j}  {bj_val}"
            tk.Label(sec2, text=line, font=FONT_MONO, bg=COL['card'], fg=COL['text']).pack(anchor='w', pady=1)

        # Segni delle variabili duali
        y_signs_str = ", ".join([f"y{i+1} {dual_info['y_signs'][i]}" for i in range(len(y_opt))])
        tk.Label(sec2, text=f"  Segni variabili duali: {y_signs_str}", font=FONT_MONO_SM, bg=COL['card'], fg=COL['primary']).pack(anchor='w', pady=(6, 0))

        # ── Sezione 3: Soluzione Ottima Duale (Prezzi Ombra) ──
        sec3 = tk.LabelFrame(f, text=" Soluzione Ottima Duale y* (Prezzi Ombra) ", font=FONT_SECTION,
                             bg=COL['card'], fg=COL['text'], bd=1, relief=tk.GROOVE, padx=12, pady=10)
        sec3.pack(fill=tk.X, pady=(0, 10))

        y_parts = [f"y{i+1}* = {self._frac(y_opt[i])}" for i in range(len(y_opt))]
        tk.Label(sec3, text=f"Vettore y* = ({', '.join(y_parts)})", font=FONT_UI_BOLD, bg=COL['card'], fg=COL['success']).pack(anchor='w', pady=(0, 4))
        tk.Label(sec3, text="Significato economico / Prezzi Ombra:\nCiascun y_i* rappresenta la variazione marginale della funzione obiettivo per una variazione unitaria del termine noto b_i (∂z*/∂b_i).",
                 font=FONT_UI, bg=COL['card'], fg=COL['text2'], justify=tk.LEFT).pack(anchor='w')

        # ── Sezione 4: Condizioni di Complementarità (Scarti Complementari) ──
        sec4 = tk.LabelFrame(f, text=" Teorema degli Scarti Complementari (Complementary Slackness) ", font=FONT_SECTION,
                             bg=COL['card'], fg='#D84315', bd=1, relief=tk.GROOVE, padx=12, pady=10)
        sec4.pack(fill=tk.X, pady=(0, 10))

        # 1. x_j * (A^T y - c)_j = 0
        tk.Label(sec4, text="1. Complementarità Variabili Primale / Vincoli Duali:  x_j · (vincolo_duale_j) = 0",
                 font=FONT_UI_BOLD, bg=COL['card'], fg=COL['text']).pack(anchor='w', pady=(0, 4))

        for j in range(self.solver.n):
            xj_val = dec_vars[j]
            # LHS vincolo duale j
            lhs_dual = sum(dual_info['A'][j, i] * y_opt[i] for i in range(len(y_opt)))
            cj_val = dual_info['b'][j]
            slack_dual = abs(lhs_dual - cj_val)
            status = "Attivo (scarto = 0)" if slack_dual < 1e-6 else f"Non attivo (scarto = {self._frac(slack_dual)})"
            compl_ok = "✔ Soddisfatta" if (abs(xj_val) < 1e-6 or slack_dual < 1e-6) else "✘"
            txt = f"   x{j+1} = {self._frac(xj_val):<6}  |  Vincolo Duale V{j+1}: {status:<26}  --> {compl_ok}"
            tk.Label(sec4, text=txt, font=FONT_MONO_SM, bg=COL['card'], fg='#2E7D32' if '✔' in compl_ok else '#C62828').pack(anchor='w')

        # 2. y_i * (b - Ax)_i = 0
        tk.Label(sec4, text="\n2. Complementarità Variabili Duali / Vincoli Primale:  y_i · (slack_primale_i) = 0",
                 font=FONT_UI_BOLD, bg=COL['card'], fg=COL['text']).pack(anchor='w', pady=(4, 4))

        for i in range(self.solver.m):
            yi_val = y_opt[i]
            lhs_prim = np.dot(self.solver.A_orig[i], dec_vars)
            bi_prim = self.solver.b_orig[i]
            slack_prim = abs(bi_prim - lhs_prim)
            status = "Attivo (scarto = 0)" if slack_prim < 1e-6 else f"Non attivo (scarto = {self._frac(slack_prim)})"
            compl_ok = "✔ Soddisfatta" if (abs(yi_val) < 1e-6 or slack_prim < 1e-6) else "✘"
            txt = f"   y{i+1} = {self._frac(yi_val):<6}  |  Vincolo Primale V{i+1}: {status:<24}  --> {compl_ok}"
            tk.Label(sec4, text=txt, font=FONT_MONO_SM, bg=COL['card'], fg='#2E7D32' if '✔' in compl_ok else '#C62828').pack(anchor='w')

    def _populate_dual_geom_tab(self, parent):
        """Costruisce la visualizzazione del poliedro duale con animazione passo-passo."""
        solver = self.solver
        animation = {"id": None}
        dual_info = solver.get_dual_formulation(is_primal_min=self.is_min)
        y_opt = solver.get_dual_solution()
        m_dual_vars = len(y_opt)  # numero di variabili duali = numero di vincoli primali

        # Controlli animazione duale in alto
        ctrl_dual = tk.Frame(parent, bg=COL['card'], bd=1, relief=tk.GROOVE, padx=10, pady=6)
        ctrl_dual.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))

        total_steps = len(solver.history)
        dual_step = tk.IntVar(value=total_steps - 1)
        animation['id'] = None

        step_label = tk.Label(ctrl_dual, text=f"Passo {dual_step.get()}/{total_steps-1} (OTTIMO)",
                                      font=FONT_UI_BOLD, bg=COL['card'], fg=COL['success'])
        step_label.pack(side=tk.LEFT, padx=(0, 12))

        def _step_dual(delta):
            cur = dual_step.get()
            nxt = max(0, min(total_steps - 1, cur + delta))
            dual_step.set(nxt)
            _render_dual_figure()

        def _stop_dual_anim():
            if animation['id'] is not None:
                parent.after_cancel(animation['id'])
                animation['id'] = None
            dual_play_btn.config(text="Play", bg=COL['succ_lt'], fg=COL['success'])

        def _dual_anim_tick():
            cur = dual_step.get()
            if cur < total_steps - 1:
                dual_step.set(cur + 1)
                _render_dual_figure()
                animation['id'] = parent.after(self.speed_ms.get(), _dual_anim_tick)
            else:
                _stop_dual_anim()

        def _toggle_dual_anim():
            if animation['id'] is not None:
                _stop_dual_anim()
            else:
                if dual_step.get() >= total_steps - 1:
                    dual_step.set(0)
                    _render_dual_figure()
                dual_play_btn.config(text="Pausa", bg='#FFE0B2', fg=COL['warning'])
                _dual_anim_tick()

        tk.Button(ctrl_dual, text="Indietro", font=FONT_UI, bg='#E0E0E0',
                  relief=tk.FLAT, cursor='hand2',
                  command=lambda: _step_dual(-1)).pack(side=tk.LEFT, padx=3)

        tk.Button(ctrl_dual, text="Avanti", font=FONT_UI, bg='#E0E0E0',
                  relief=tk.FLAT, cursor='hand2',
                  command=lambda: _step_dual(1)).pack(side=tk.LEFT, padx=3)

        dual_play_btn = tk.Button(ctrl_dual, text="Play", font=FONT_UI_BOLD,
                                  bg=COL['succ_lt'], fg=COL['success'],
                                  relief=tk.FLAT, cursor='hand2',
                                  command=_toggle_dual_anim)
        dual_play_btn.pack(side=tk.LEFT, padx=(10, 3))

        fig_dual = Figure(figsize=(8, 5.5), dpi=100, facecolor=COL['card'])
        canvas_dual = FigureCanvasTkAgg(fig_dual, master=parent)
        canvas_dual.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        from geometry import vertices, draw_region
        M_bound = max(float(np.max(np.abs(y_opt))) * 2.5, 5.0)
        A_d, b_d, signs_d = list(dual_info['A']), list(dual_info['b']), list(dual_info['constr_signs'])
        for i, sign in enumerate(dual_info['y_signs']):
            if sign != 'libera':
                row = np.zeros(m_dual_vars); row[i] = 1
                A_d.append(row); b_d.append(0); signs_d.append('>=' if sign == '>= 0' else '<=')
        dual_verts = vertices(A_d, b_d, signs_d, m_dual_vars, nonnegative=False,
                              bounds=[(-M_bound, M_bound)] * m_dual_vars) if m_dual_vars in (2, 3) else []
        dual_verts_3d = dual_verts

        def _render_dual_figure():
            fig_dual.clear()
            s_idx = dual_step.get()
            y_curr = solver.get_dual_vector_at_step(s_idx)

            # Aggiorna label stato
            if s_idx == total_steps - 1:
                step_label.config(
                    text=f"Passo {s_idx}/{total_steps-1} (OTTIMO)",
                    fg=COL['success'])
            else:
                step_label.config(
                    text=f"Passo {s_idx}/{total_steps-1}",
                    fg=COL['text'])

            # Calcola traiettoria duale fino a s_idx
            dual_traj = [y for k in range(s_idx + 1) if (y := solver.get_dual_vector_at_step(k)) is not None]
            traj_arr = np.array(dual_traj).reshape(-1, m_dual_vars)

            if m_dual_vars == 2:
                ax = fig_dual.add_subplot(111)
                max_val = M_bound
                draw_region(ax, dual_verts, .14, color="#AB47BC")
                y1_range = np.linspace(-max_val, max_val, 300)

                colors_d = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#00838F']
                for j in range(len(dual_info['b'])):
                    a1 = dual_info['A'][j, 0]
                    a2 = dual_info['A'][j, 1]
                    bj = dual_info['b'][j]
                    sign_j = dual_info['constr_signs'][j]
                    c_line = colors_d[j % len(colors_d)]

                    if abs(a2) > 1e-10:
                        y2_line = (bj - a1 * y1_range) / a2
                        mask = (y2_line >= -max_val) & (y2_line <= max_val)
                        lbl = f"{self._frac(a1)}y₁+{self._frac(a2)}y₂ {sign_j} {self._frac(bj)}"
                        ax.plot(y1_range[mask], y2_line[mask], color=c_line, linewidth=1.5, label=lbl)
                    elif abs(a1) > 1e-10:
                        y1_val = bj / a1
                        ax.axvline(x=y1_val, color=c_line, linewidth=1.5, label=f"y₁ {sign_j} {self._frac(y1_val)}")

                # Traiettoria duale
                if len(traj_arr) > 1:
                    ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'm--', linewidth=2.2, alpha=0.8, zorder=8)

                for k, pt in enumerate(traj_arr):
                    if k == 0:
                        ax.scatter(*pt[:2], c=COL['vertex_start'], s=120, marker='^', edgecolors='black', zorder=9, label='Start Duale')
                    elif k == s_idx and s_idx == total_steps - 1:
                        ax.scatter(*pt[:2], c=COL['vertex_optimal'], s=220, marker='*', edgecolors='black', zorder=12, label='Ottimo y*')
                    elif k == s_idx:
                        ax.scatter(*pt[:2], c=COL['vertex_current'], s=110, marker='D', edgecolors='black', zorder=10, label='Corrente')
                    else:
                        ax.scatter(*pt[:2], c='#CE93D8', s=50, marker='o', edgecolors='black', zorder=7)

                ax.set_xlabel("y₁ (Variabile Duale 1)", fontsize=11, fontweight='bold', color=COL['primary'])
                ax.set_ylabel("y₂ (Variabile Duale 2)", fontsize=11, fontweight='bold', color=COL['primary'])
                ax.set_title(f"Evoluzione del Vettore Duale y — Iterazione {s_idx}", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9, loc='upper right')
                ax.set_xlim([-max_val, max_val])
                ax.set_ylim([-max_val, max_val])

            elif m_dual_vars == 3:
                ax = fig_dual.add_subplot(111, projection='3d')

                draw_region(ax, dual_verts_3d, .16, color="#AB47BC")

                # Traiettoria duale
                if len(traj_arr) > 1:
                    ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
                            color='#8E24AA', linestyle='--', linewidth=2.2, alpha=0.85, zorder=8)

                for k, pt in enumerate(traj_arr):
                    if k == 0:
                        ax.scatter(*pt[:3], c=COL['vertex_start'], s=120, marker='^', edgecolors='black', zorder=9, label='Start Duale')
                    elif k == s_idx and s_idx == total_steps - 1:
                        ax.scatter(*pt[:3], c=COL['vertex_optimal'], s=240, marker='*', edgecolors='black', zorder=12, label='Ottimo y*')
                    elif k == s_idx:
                        ax.scatter(*pt[:3], c=COL['vertex_current'], s=120, marker='D', edgecolors='black', zorder=10, label='Corrente y')
                    else:
                        ax.scatter(*pt[:3], c='#E1BEE7', s=50, marker='o', edgecolors='purple', zorder=7)

                    # Etichetta valore a ogni passo visitato
                    if k == s_idx:
                        y_str = f"y=({self._frac(pt[0])}, {self._frac(pt[1])}, {self._frac(pt[2])})"
                        ax.text(pt[0] + 0.08, pt[1] + 0.08, pt[2] + 0.08, y_str,
                                fontsize=8, fontweight='bold', color='#4A148C',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='gray'))

                ax.set_xlabel("y₁", fontsize=11, fontweight='bold', color=COL['primary'])
                ax.set_ylabel("y₂", fontsize=11, fontweight='bold', color=COL['primary'])
                ax.set_zlabel("y₃", fontsize=11, fontweight='bold', color=COL['primary'])
                ax.set_title(f"Regione Ammissibile Duale e Percorso y — Iterazione {s_idx}", fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, loc='upper left')

                max_v = M_bound
                ax.set_xlim([-max_v, max_v])
                ax.set_ylim([-max_v, max_v])
                ax.set_zlim([-max_v, max_v])
            else:
                ax = fig_dual.add_subplot(111)
                subs = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
                m_sub = str(m_dual_vars).translate(subs)
                ax.text(0.5, 0.5, f"Il problema duale ha {m_dual_vars} variabili (y₁ ... y{m_sub}).\nLa visualizzazione geometrica cartesiana è disponibile per 2 o 3 variabili.\n\nConsulta la scheda 'Formulazione & Scarti Complementari'\nper tutti i dettagli algebrici esatti.",
                        ha='center', va='center', fontsize=13, color=COL['text2'], transform=ax.transAxes)
                ax.set_axis_off()

            fig_dual.tight_layout()
            canvas_dual.draw()

        def _destroy_dual(event):
            if event.widget is parent and animation['id'] is not None:
                self.root.after_cancel(animation['id'])
                animation['id'] = None
        parent.bind('<Destroy>', _destroy_dual, add='+')

        # Render iniziale
        _render_dual_figure()


# ═══════════════════════════════════════════════════════
#  Entry point con Splash Screen
# ═══════════════════════════════════════════════════════

def show_splash(root):
    """Mostra una splash screen elegante e centrata all'avvio dell'applicazione."""
    splash_path = asset_path('splash.png')
    if not (HAS_PIL and os.path.exists(splash_path)):
        return None

    try:
        splash = tk.Toplevel(root)
        splash.overrideredirect(True)  # Rimuove bordi e barra del titolo della finestra

        img_raw = Image.open(splash_path)
        # Dimensioni splash screen: 640x360 (proporzione 16:9 esatta)
        splash_w, splash_h = 640, 360
        img_resized = img_raw.resize((splash_w, splash_h), Image.Resampling.LANCZOS)
        splash_img = ImageTk.PhotoImage(img_resized)

        # Centra la splash screen sullo schermo
        sw = splash.winfo_screenwidth()
        sh = splash.winfo_screenheight()
        x = (sw - splash_w) // 2
        y = (sh - splash_h) // 2
        splash.geometry(f"{splash_w}x{splash_h}+{x}+{y}")

        lbl = tk.Label(splash, image=splash_img, bd=0, highlightthickness=0)
        lbl.image = splash_img  # mantieni riferimento per evitare garbage collection
        lbl.pack(fill=tk.BOTH, expand=True)

        splash.update()
        return splash
    except Exception:
        return None

def main():
    root = tk.Tk()
    root.withdraw()  # Nascondi la finestra principale durante la splash screen

    splash = show_splash(root)

    # Costruisci l'interfaccia principale in background
    app = SimplexGUI(root)

    def _close_splash_and_show():
        if splash:
            try:
                splash.destroy()
            except Exception:
                pass
        root.deiconify()  # Mostra la finestra principale
        try:
            root.state('zoomed')  # Avvia a schermo intero massimizzato
        except Exception:
            pass
        root.lift()
        root.focus_force()

    # Tempo di visualizzazione della splash screen: 1.5 secondi (abbastanza per ammirarla, senza far aspettare)
    if splash:
        root.after(1500, _close_splash_and_show)
    else:
        root.deiconify()
        try:
            root.state('zoomed')
        except Exception:
            pass

    root.mainloop()

if __name__ == '__main__':
    main()
