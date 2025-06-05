# custom_schieber_env_flat.py

import gym
import numpy as np
from gym import spaces

from schieber.game import Game
from schieber.player.external_player import ExternalPlayer
from schieber.player.greedy_player.greedy_player import GreedyPlayer
from schieber.player.random_player import RandomPlayer
from schieber.team import Team
from schieber.card import Card, from_string_to_index, from_card_to_index
from schieber.trumpf import Trumpf as TrumpfEnum

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS: must exactly match your HDF5 layout
NUM_TRUMPS = 1     # one slot for current trumpf (value 0..3)
NUM_CARDS_HAND = 9     # RL agent’s hand slots
NUM_CARDS_TABLE = 4     # cards on the current trick’s table
NUM_CARDS_HISTORY = 32    # last 32 played cards
# other “seen” cards (played or on table but not in hand)
NUM_CARDS_SHOWN = 27
NUM_STATE = NUM_TRUMPS + NUM_CARDS_HAND + \
    NUM_CARDS_TABLE + NUM_CARDS_HISTORY + NUM_CARDS_SHOWN
# Total = 1 + 9 + 4 + 32 + 27 = 73 (adjust if your HDF5 differs)
# ──────────────────────────────────────────────────────────────────────────────


class SchieberEnvFlat(gym.Env):
    """
    Gym environment for Swiss Jass (Schieber), returning a flat integer vector of length NUM_STATE.

    Observation layout (length=NUM_STATE):
      [0]                              = trumpf ∈ {0..3}
      [1..9]                           = RL agent’s 9 hand slots (0..35 or 36 if empty)
      [10..13]                         = current trick’s 4 table slots (0..35 or 36 if empty)
      [14..45]                         = last 32 played cards (0..35 or 36 if none)
      [46..72]                         = shown cards (0..35 or 36 if none)

    Action space: Discrete(9) → choose one card from your 9-card hand.

    Reward: 0 until someone reaches point_limit; when game ends, reward = (team0_points − team1_points).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, point_limit: int = 2500, seed: int = None):
        super().__init__()

        # 1) Build players: index 0 = RL, index 1 = Opp1, index 2 = Partner, index 3 = Opp2
        self.rl_player = ExternalPlayer(name="RL-Agent")
        partner = GreedyPlayer(name="Partner")
        opp1 = RandomPlayer(name="Opp1")
        opp2 = RandomPlayer(name="Opp2")
        teams = [Team([self.rl_player, partner]), Team([opp1, opp2])]

        # 2) Create the Game
        self.game = Game(teams, point_limit=point_limit)
        if seed is not None:
            self.game.seed = seed

        # 3) Define Gym spaces
        #    - slot 0: trumpf (0..3) → use `4` in MultiDiscrete
        #    - each “card slot”: 0..36, where 36 = “empty” → use `37` in MultiDiscrete
        disc_list = [4] \
            + [37] * NUM_CARDS_HAND \
            + [37] * NUM_CARDS_TABLE \
            + [37] * NUM_CARDS_HISTORY \
            + [37] * NUM_CARDS_SHOWN
        self.observation_space = spaces.MultiDiscrete(disc_list)

        # 4) Action: pick one of 9 positions in hand
        self.action_space = spaces.Discrete(NUM_CARDS_HAND)

    def reset(self):
        """
        Start a brand‐new hand:
          1) Clear points & past tricks (self.game.reset()).
          2) Shuffle and deal cards (self.game.dealer.shuffle_cards/ deal_cards).
          3) Choose trumpf (self.game.define_trumpf(start_player_index=0)).
          4) Return the encoded flat state.
        """
        # 1) Clear previous hand’s data
        self.game.reset()

        # 2) Shuffle & deal
        self.game.dealer.shuffle_cards(self.game.seed)
        self.game.dealer.deal_cards()

        # 3) Choose trumpf with starting index = 0
        #    This sets self.game.trumpf to a valid TrumpfEnum (no longer None)
        self.game.define_trumpf(start_player_index=0)

        # 4) Encode and return
        status = self.game.get_status()
        return self._encode_flat(status)

    def step(self, action: int):
        """
        Play one full trick (stich) starting from RL’s turn:
          1) Map action index → chosen_card from self.rl_player.cards.
          2) Call self.game.play_stich(...) to play that trick.
          3) Count points for that stich (self.game.count_points(...)).
          4) Check if any team has reached point_limit → done + reward.
          5) Fetch new status, encode, and return (next_state, reward, done, {}).
        """
        # 1) Pick the actual Card object from RL’s hand
        hand = self.rl_player.cards
        assert 0 <= action < len(
            hand), f"Action {action} invalid; hand size = {len(hand)}"
        chosen_card = hand[action]

        # 2) Play the trick. We need RL’s index in self.game.players
        rl_idx = self.game.players.index(self.rl_player)
        stich = self.game.play_stich(start_player_index=rl_idx)

        # 3) Count points for that stich; if it was 9th stich, mark last=True
        # After 8 prior stiches, the 9th is “last”
        last_flag = (len(self.game.stiche) == 8)
        self.game.count_points(stich, last=last_flag)

        # 4) Check for terminal
        done = (
            self.game.teams[0].won(self.game.point_limit) or
            self.game.teams[1].won(self.game.point_limit)
        )
        if done:
            pts0 = self.game.teams[0].points
            pts1 = self.game.teams[1].points
            reward = float(pts0 - pts1)
        else:
            reward = 0.0

        # 5) Encode next state
        status = self.game.get_status()
        next_state = self._encode_flat(status)
        return next_state, reward, done, {}

    def render(self, mode="human"):
        """
        Print current trumpf and team scores.
        """
        status = self.game.get_status()
        trumpf = status["trumpf"]
        pts0 = status["teams"][0]["points"]
        pts1 = status["teams"][1]["points"]
        print(f"Trumpf = {trumpf} | Team0 pts = {pts0} | Team1 pts = {pts1}")

    def close(self):
        pass

    def _encode_flat(self, status: dict) -> np.ndarray:
        """
        Convert Game.get_status() → flat np.ndarray (shape = NUM_STATE, dtype=int64).

        status keys:
          • 'stiche'   : list of past tricks (each has 'played_cards': a list of {'card': "<suit+rank>"})
          • 'trumpf'   : a string like "OBEABE", "UNDEUFE", etc.
          • 'table'    : list of dicts {'card': "<suit+rank>"} for the current trick (0..4)
          • 'teams'    : [ {'points': int}, {'points': int} ]
          • … (we ignore 'geschoben', 'point_limit')

        We fill:
          state[0]                              = TrumpfEnum[status['trumpf']].value  # 0..3
          state[1..9]                           = RL’s hand (0..35 or 36 if empty)
          state[10..13]                         = current trick’s table (0..35 or 36 if empty)
          state[14..45]                         = last 32 played cards (0..35 or 36 if none)
          state[46..72]                         = shown cards (0..35 or 36 if none)
        """
        arr = np.full((NUM_STATE,), 36, dtype=np.int64)  # default “empty”=36

        # 1) Trumpf → index 0
        trumpf_str = status["trumpf"]                   # e.g. "OBEABE"
        trumpf_val = TrumpfEnum[trumpf_str].value       # e.g. 0..3
        arr[0] = int(trumpf_val)

        # 2) RL’s hand → slots 1..9
        hand_cards = self.rl_player.cards
        for i in range(NUM_CARDS_HAND):
            if i < len(hand_cards):
                cidx = from_card_to_index(hand_cards[i])      # 0..35
                arr[1 + i] = int(cidx)
            else:
                arr[1 + i] = 36                           # empty

        # 3) Table → slots 10..13
        table_cards = status.get("table", [])
        for j in range(NUM_CARDS_TABLE):
            if j < len(table_cards):
                cstr = table_cards[j]["card"]
                cidx = from_string_to_index(cstr)        # 0..35
                arr[1 + NUM_CARDS_HAND + j] = int(cidx)
            else:
                arr[1 + NUM_CARDS_HAND + j] = 36

        # 4) History → slots 14..45 (32 cards)
        played = []
        for trick in status.get("stiche", []):
            for entry in trick.get("played_cards", []):
                cstr = entry["card"]
                cidx = from_string_to_index(cstr)
                played.append(int(cidx))
        if len(played) >= NUM_CARDS_HISTORY:
            recent = played[-NUM_CARDS_HISTORY:]
        else:
            recent = [36] * (NUM_CARDS_HISTORY - len(played)) + played
        for h in range(NUM_CARDS_HISTORY):
            arr[1 + NUM_CARDS_HAND + NUM_CARDS_TABLE + h] = int(recent[h])

        in_hand = {from_card_to_index(c) for c in hand_cards}
        on_table = {from_string_to_index(e["card"]) for e in table_cards}
        played_set = set(played)
        shown_set = (played_set | on_table) - in_hand
        shown_sorted = sorted(shown_set)
        if len(shown_sorted) >= NUM_CARDS_SHOWN:
            final_shown = shown_sorted[:NUM_CARDS_SHOWN]
        else:
            final_shown = shown_sorted + [36] * \
                (NUM_CARDS_SHOWN - len(shown_sorted))
        for s in range(NUM_CARDS_SHOWN):
            arr[1 + NUM_CARDS_HAND + NUM_CARDS_TABLE +
                NUM_CARDS_HISTORY + s] = int(final_shown[s])

        return arr
