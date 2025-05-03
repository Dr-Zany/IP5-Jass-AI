import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import h5py
import os

# Card mappings
RANKS = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['Diamonds', 'Clubs', 'Hearts', 'Spades']
# Trump unicode symbols
TRUMP = ['♦', '♣', '♥', '♠', '↓', '↑']  # ♦ ♣ ♥ ♠ ↓ ↑
TRUMP_NAMES = ['Diamonds', 'Clubs', 'Hearts', 'Spades', 'Bottom-up', 'Top-down']

# Build deck list: 36 cards
DECK = [f"{rank} of {suit}" for suit in SUITS for rank in RANKS]

# Directory where individual card images are stored
CARD_ASSET_DIR = os.path.join('assets', 'cards')
BACK_ASSET = os.path.join('assets', 'back.png')

# Choose resampling filter for resize
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS

class StateViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HDF5 Game State Viewer")
        self.geometry("830x700")
        self.filepath = None
        self.h5file = None
        self.card_images = {}    # mapping card name -> PhotoImage
        self.back_image = None

        self.load_assets()
        self.create_widgets()

        self.focus_set()
        self.bind_all('<Left>', lambda e: self.navigate(-1))
        self.bind_all('<Right>', lambda e: self.navigate(1))

    def load_assets(self):
        # Load card face images
        for card in DECK:
            fname = card.replace(' of ', '_of_') + '.png'
            path = os.path.join(CARD_ASSET_DIR, fname)
            if os.path.exists(path):
                img = Image.open(path).resize((80, 120), RESAMPLE_FILTER)
                self.card_images[card] = ImageTk.PhotoImage(img)
            else:
                self.card_images[card] = None
        # Load back image
        if os.path.exists(BACK_ASSET):
            img = Image.open(BACK_ASSET).resize((80, 120), RESAMPLE_FILTER)
            self.back_image = ImageTk.PhotoImage(img)

    def create_widgets(self):
        # Top controls
        control = ttk.Frame(self, padding=10)
        control.pack(fill=tk.X)

        ttk.Button(control, text="Open HDF5...", command=self.open_file).pack(side=tk.LEFT)
        self.game_var = tk.StringVar()
        self.game_menu = ttk.OptionMenu(control, self.game_var, "Select game...")
        self.game_menu.pack(side=tk.LEFT, padx=5)

        ttk.Label(control, text="State Index:").pack(side=tk.LEFT, padx=(10,0))
        self.idx_var = tk.IntVar(value=0)
        self.idx_spin = ttk.Spinbox(control, from_=0, to=0, textvariable=self.idx_var, width=5)
        self.idx_spin.pack(side=tk.LEFT, padx=5)

        ttk.Button(control, text="Load State", command=self.load_state).pack(side=tk.LEFT, padx=5)

        # Scrollable display area
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container)
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.display_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0,0), window=self.display_frame, anchor='nw')

        # Bind scroll region update and mouse wheel
        self.display_frame.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind_all('<Button-4>', lambda e: self.canvas.yview_scroll(-1, 'units'))
        self.canvas.bind_all('<Button-5>', lambda e: self.canvas.yview_scroll(1, 'units'))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 120), 'units')

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*")])
        if not path:
            return
        try:
            self.h5file = h5py.File(path, 'r')
            games = list(self.h5file.keys())
            menu = self.game_menu['menu']
            menu.delete(0, 'end')
            for g in games:
                menu.add_command(label=g, command=tk._setit(self.game_var, g, self.on_game_selected))
            if games:
                self.game_var.set(games[0])
                self.on_game_selected()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file:\n{e}")

    def on_game_selected(self, *args):
        grp = self.h5file.get(self.game_var.get())
        if grp and 'state' in grp and 'action' in grp:
            max_idx = grp['state'].shape[0] - 1
            self.idx_spin.config(to=max_idx)
            self.idx_var.set(0)
        else:
            messagebox.showwarning("Warning", "Selected game must have 'state' and 'action' datasets.")

    def load_state(self):
        game = self.game_var.get()
        idx = self.idx_var.get()
        try:
            grp = self.h5file[game]
            state_vec = grp['state'][idx]
            action_idx = int(grp['action'][idx][0])  # absolute card index played
            decoded = self.decode_state(state_vec)
            decoded['action'] = action_idx
            self.show_cards(decoded)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load state or action:\n{e}")

    def navigate(self, delta):
        # Move index and reload state
        try:
            current = self.idx_var.get()
            new = current + delta
            if 0 <= new <= int(self.idx_spin.cget('to')):
                self.idx_var.set(new)
                self.load_state()
        except Exception:
            pass

    def decode_state(self, vec):
        history = [int(v) - 1 for v in vec[0:32]]
        played = [int(v) - 1 for v in vec[32:35]]
        hand = [int(v) - 1 for v in vec[35:44]]
        others = []
        for p in range(3):
            start = 44 + p*9
            others.append([int(v) - 1 for v in vec[start:start+9]])
        trump_idx = int(vec[71]) -1 if len(vec) > 71 else None
        return {
            'history': history,
            'played': played,
            'hand': hand,
            'others': others,
            'trump_idx': trump_idx
        }

    def show_cards(self, d):
        # Clear previous
        for child in self.display_frame.winfo_children():
            child.destroy()

        def add_row(title, indices, frame, highlight=None):
            lbl = ttk.Label(frame, text=title)
            lbl.pack(side=tk.TOP, anchor='w')
            row = ttk.Frame(frame)
            row.pack(side=tk.TOP, pady=5)
            for idx in indices:
                card_name = DECK[idx] if 0 <= idx < len(DECK) else None
                img = self.card_images.get(card_name, self.back_image)
                l = ttk.Label(row, image=img)
                l.image = img
                if highlight is not None and idx == highlight:
                    l.config(borderwidth=10, relief='solid', background='yellow')
                l.pack(side=tk.LEFT, padx=2)

        # History in 4 rows
        hist_frame = ttk.LabelFrame(self.display_frame, text='History')
        hist_frame.pack(fill=tk.X, padx=5, pady=5)
        for i in range(4):
            slice_ = d['history'][i*8:(i+1)*8]
            add_row("", slice_, hist_frame)

        # Played
        play_frame = ttk.LabelFrame(self.display_frame, text='Current Played')
        play_frame.pack(fill=tk.X, padx=5, pady=5)
        add_row('', d['played'], play_frame)

        # Trump
        trump_frame = ttk.Frame(self.display_frame)
        trump_frame.pack(pady=10)
        if d['trump_idx'] is not None and 0 <= d['trump_idx'] < len(TRUMP):
            symbol = TRUMP[d['trump_idx']]
            name = TRUMP_NAMES[d['trump_idx']]
            lbl = ttk.Label(trump_frame, text=f"Trump: {symbol} ({name})", font=(None, 16))
            lbl.pack()

        # Hand
        hand_frame = ttk.LabelFrame(self.display_frame, text='Your Hand')
        hand_frame.pack(fill=tk.X, padx=5, pady=5)
        add_row('', d['hand'], hand_frame, highlight=d['hand'][d['action']])

        # Others
        for i, player in enumerate(d['others'], start=1):
            oframe = ttk.LabelFrame(self.display_frame, text=f'Player {i} Cards')
            oframe.pack(fill=tk.X, padx=5, pady=5)
            add_row("", player, oframe)

if __name__ == '__main__':
    app = StateViewerApp()
    app.mainloop()
