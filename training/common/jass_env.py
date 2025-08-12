import random
from typing import List, Tuple
from dataclasses import dataclass

import torch
from torch import tensor
from torch.nn import functional as F

from .model_dnn import ModelDNN

NUM_CARDS = 36
TRUMP_MODES = {
    1: "Diamonds",
    2: "Hearts",
    3: "Spades",
    4: "Clubs",
    5: "TopDown",
    6: "BottomUp",
}


def decode_card(card: int) -> Tuple[int, int]:
    suit = (card - 1) // 9
    rank = (card - 1) % 9
    return suit, rank


def is_trump(card: int, trump: int) -> bool:
    suit, _ = decode_card(card)
    return trump in (1, 2, 3, 4) and suit == (trump - 1)


class JassModel:
    def __init__(self, model_jass: ModelDNN, model_trump: ModelDNN, device: str = 'cpu'):
        self.model_jass = model_jass.to(device)
        self.model_trump = model_trump.to(device)
        self.device = device

    @torch.no_grad()
    def act(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:    [B, 72]  long
        mask: [B, 9]   bool/0-1
        returns: (idx:[B], action:[B,9])
        """
        x = x.to(self.device)
        mask = mask.to(self.device)
        logits = self.model_jass(x)                 # [B,9]
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        actions = F.gumbel_softmax(masked_logits, tau=1.0, hard=True)  # [B,9]
        idx = actions.argmax(dim=-1)                # [B]
        return idx, actions

    @torch.no_grad()
    def choose_trump_batch(self, hands: torch.Tensor, must_choose: torch.Tensor) -> torch.Tensor:
        """
        hands:       [B, 9] long (0-padded not needed for trump; game deals 9)
        must_choose: [B]    long {0,1}
        returns trump choices: [B] long in {0..6}
        """
        x = torch.cat([hands, must_choose.view(-1,1)], dim=1).to(self.device)  # [B,10]
        logits = self.model_trump(x)  # [B,7]

        mask = torch.ones((hands.size(0), 7), dtype=torch.float, device=self.device)
        # disable pass where must_choose==1
        mask[must_choose.to(self.device) == 1, 0] = 0.0

        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        choice = F.gumbel_softmax(masked_logits, tau=1.0, hard=True).argmax(dim=-1)  # [B]
        return choice





class JassState:
    def __init__(self, history: List[int], table: List[int], hand: List[int], gewisen: List[int], trump: int):
        self.history = history
        self.table = table
        self.hand = hand
        self.gewisen = gewisen
        self.trump = trump

    def make_tensor(self) -> tensor:
        return tensor(
            self.history + self.table + self.hand + self.gewisen + [self.trump],
            dtype=torch.long
        ).view(1, -1)  # [1, 72]


class TrumpChoiceState:
    def __init__(self, hand: List[int], must_choose: int):
        self.hand = hand
        self.must_choose = must_choose

    def make_tensor(self) -> tensor:
        return tensor(self.hand + [self.must_choose], dtype=torch.long).view(1, -1)


def _pad_list(lst, target_len, pad=0):
    return lst + [pad] * (target_len - len(lst))

def _pack_state_batch(histories, tables, hands, gewisens, trumps) -> torch.Tensor:
    """
    histories:  list[B] of list[int] (0..32 already padded/trimmed here)
    tables:     list[B] of list[int] (0..3 padded)
    hands:      list[B] of list[int] (<=9 padded)
    gewisens:   list[B] of list[int] (len 27)
    trumps:     list[B] of int
    returns: [B,72] long
    """
    B = len(hands)
    rows = []
    for b in range(B):
        row = (
            _pad_list(histories[b], 32, 0) +
            _pad_list(tables[b], 3, 0) +
            _pad_list(hands[b], 9, 0) +
            gewisens[b][:] +             # already 27
            [trumps[b]]
        )
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)

def _legal_mask_batch(env, hands, tables, trumps) -> torch.Tensor:
    """
    hands:  list[B] of list[int] (actual current hands)
    tables: list[B] of list[int]
    trumps: list[B] of int
    returns: [B,9] bool
    """
    B = len(hands)
    mask = torch.zeros((B, 9), dtype=torch.bool)
    for b in range(B):
        m = env._legal_card_mask(hands[b], tables[b], trumps[b])  # [9] bool
        mask[b] = m
    return mask


class JassEnv:
    def __init__(self):
        self.trump_caller = 0

    def play_game(self, model: JassModel, B: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs B games in parallel with batched policy evaluation.
        Returns:
          S: [B, 36, 72] long
          A: [B, 36,  9] float
        """
        device = model.device

        # outputs
        S = torch.zeros((B, 36, 72), dtype=torch.long, device=device)
        A = torch.zeros((B, 36, 9), dtype=torch.float, device=device)

        # ----- deal -----
        decks = [list(range(1, NUM_CARDS + 1)) for _ in range(B)]
        for d in decks: random.shuffle(d)
        hands = [[sorted(decks[b][i * 9:(i + 1) * 9]) for i in range(4)] for b in range(B)]
        initial_hands = [[h[:] for h in hands[b]] for b in range(B)]

        # ----- batch trump choice -----
        callers = [self.trump_caller for _ in range(B)]
        must_choose0 = torch.zeros(B, dtype=torch.long)
        hands_caller = torch.tensor([hands[b][callers[b]] for b in range(B)], dtype=torch.long)
        choice = model.choose_trump_batch(hands_caller, must_choose0)  # [B]

        # second pass where pass==0 occurred
        need_partner = (choice == 0)
        if need_partner.any():
            partners = torch.tensor([(c + 2) % 4 for c in callers], dtype=torch.long)
            hands_partner = torch.tensor(
                [hands[b][partners[b].item()] for b in range(B)], dtype=torch.long
            )
            must_choose1 = torch.ones(B, dtype=torch.long)
            choice2 = model.choose_trump_batch(hands_partner, must_choose1)
            # apply only for those who passed
            choice = torch.where(need_partner, choice2, choice)

        if (choice == 0).any():
            raise ValueError("No trump chosen for at least one game.")

        trumps = choice.tolist()
        # next leader after caller (same as single game), and advance class-level caller once
        current_leaders = [ (callers[b] + 1) % 4 for b in range(B) ]
        self.trump_caller = (self.trump_caller + 1) % 4

        # per-batch runtime state
        histories = [[] for _ in range(B)]
        tables    = [[] for _ in range(B)]
        gewisen_all = [[[0] * 27 for _ in range(4)] for _ in range(B)]
        has_revealed_weisen = [[False] * 4 for _ in range(B)]

        # ----- 9 tricks → 36 steps -----
        step = 0
        for trick_i in range(9):
            tricks = [[] for _ in range(B)]
            trick_players = [[] for _ in range(B)]

            for turn in range(4):
                # who plays this turn (per env)
                players = [ (current_leaders[b] + turn) % 4 for b in range(B) ]

                # build batched state components
                hands_curr = [hands[b][players[b]] for b in range(B)]
                x = _pack_state_batch(
                    histories=[_pad_list(histories[b], 32, 0) for b in range(B)],
                    tables=[_pad_list(tables[b], 3, 0) for b in range(B)],
                    hands=hands_curr,
                    gewisens=[gewisen_all[b][players[b]][:] for b in range(B)],
                    trumps=trumps
                ).to(device)  # [B,72]

                # batched legal mask [B,9]
                mask = _legal_mask_batch(self, hands_curr, tables, trumps).to(device)

                # single batched policy call
                idx, a = model.act(x, mask)  # idx:[B], a:[B,9]

                # log outputs
                S[:, step, :] = x
                A[:, step, :] = a.to(torch.float)
                step += 1

                # apply actions per env
                for b in range(B):
                    player_i = players[b]
                    slot = int(idx[b].item())
                    chosen_card = hands[b][player_i].pop(slot)
                    tables[b].append(chosen_card)
                    tricks[b].append(chosen_card)
                    trick_players[b].append(player_i)

                    # reveal weisen after this player's first play
                    if not has_revealed_weisen[b][player_i]:
                        has_revealed_weisen[b][player_i] = True
                        weise_cards = self._detect_weisen(initial_hands[b][player_i], trump=trumps[b])
                        if weise_cards:
                            for viewer in range(4):
                                if viewer == player_i: continue
                                relative_pos = (player_i - viewer - 1) % 4
                                if relative_pos >= 3: continue
                                base_idx = relative_pos * 9
                                seg = weise_cards[:9] + [0] * (9 - min(9, len(weise_cards)))
                                gewisen_all[b][viewer][base_idx:base_idx + 9] = seg

            # resolve trick winners per env
            for b in range(B):
                winner = self._determine_trick_winner(tricks[b], trick_players[b], trumps[b])
                histories[b].extend(tricks[b])
                tables[b].clear()
                current_leaders[b] = winner

        return S.to(torch.long), A.to(torch.float)




    # ---------- Rules ----------
    def _detect_weisen(self, hand: List[int], trump: int = 0, scoring: dict | None = None) -> List[int]:
        from collections import defaultdict
        if scoring is None:
            scoring = {"run": {3: 20, 4: 50, "ge5_base": 100, "ge5_step": 20}, "four_kind": {"J": 200, "9": 150, "other": 100}}
        def rank_to_name(r: int) -> str:
            names = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
            return names[r]
        suits = defaultdict(list)
        ranks = defaultdict(list)
        for card in hand:
            suit, rank = decode_card(card)
            suits[suit].append(rank)
            ranks[rank].append(card)
        candidates = []
        for suit_id, ranks_in_suit in suits.items():
            s = sorted(set(ranks_in_suit))
            if not s:
                continue
            start = s[0]
            prev = s[0]
            for r in s[1:] + [None]:
                if r is not None and r == prev + 1:
                    prev = r
                    continue
                run_len = prev - start + 1
                if run_len >= 3:
                    run_cards = [suit_id * 9 + rr + 1 for rr in range(start, prev + 1)]
                    score = scoring["run"][run_len] if run_len < 5 else scoring["run"]["ge5_base"] + scoring["run"]["ge5_step"] * (run_len - 5)
                    candidates.append({"type": "run", "cards": run_cards, "score": score, "meta": {"len": run_len, "suit": suit_id, "top_rank": prev}})
                if r is None:
                    break
                start = prev = r
        for rank, cards_of_rank in ranks.items():
            if len({decode_card(c)[0] for c in cards_of_rank}) == 4:
                name = rank_to_name(rank)
                score = scoring["four_kind"]["J"] if name == "J" else scoring["four_kind"]["9"] if name == "9" else scoring["four_kind"]["other"]
                four_cards = [s * 9 + rank + 1 for s in range(4)]
                candidates.append({"type": "four", "cards": four_cards, "score": score, "meta": {"rank": rank}})
        if not candidates:
            return []
        def tie_key(c):
            if c["type"] == "run":
                trump_bonus = 1 if (trump in (1, 2, 3, 4) and (c["meta"]["suit"] == trump - 1)) else 0
                return (c["score"], len(c["cards"]), c["meta"]["top_rank"], trump_bonus, -c["meta"]["suit"])
            else:
                return (c["score"], len(c["cards"]), c["meta"]["rank"], 0, 0)
        best = max(candidates, key=tie_key)
        return best["cards"]

    def _legal_card_mask(self, hand: List[int], table: List[int], trump: int) -> tensor:
        mask = torch.zeros(9, dtype=torch.bool)
        for i, card in enumerate(hand + [0] * (9 - len(hand))):
            if card != 0 and self._is_valid_play(card, hand, table, trump):
                mask[i] = True
        return mask

    def _is_valid_play(self, card: int, hand: List[int], table: List[int], trump: int) -> bool:
        if not table:
            return True
        lead_suit, _ = decode_card(table[0])
        card_suit, _ = decode_card(card)
        has_lead = any(decode_card(c)[0] == lead_suit for c in hand)
        has_trump = any(is_trump(c, trump) for c in hand) if trump in (1, 2, 3, 4) else False
        if has_lead:
            return card_suit == lead_suit
        if trump in (1, 2, 3, 4) and has_trump:
            return is_trump(card, trump)
        return True

    def _determine_trick_winner(self, trick: List[int], players: List[int], trump: int) -> int:
        TRUMP_WEIGHTS = [0, 1, 2, 7, 3, 8, 4, 5, 6]
        NONTRUMP_TOP = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        NONTRUMP_BOTTOM = [8, 7, 6, 5, 4, 3, 2, 1, 0]
        lead_suit, _ = decode_card(trick[0])
        def power(card: int) -> tuple:
            suit, rank = decode_card(card)
            if trump in (1, 2, 3, 4):
                trump_suit = trump - 1
                if suit == trump_suit:
                    return (2, TRUMP_WEIGHTS[rank])
                elif suit == lead_suit:
                    return (1, NONTRUMP_TOP[rank])
                else:
                    return (0, -1)
            elif trump == 5:
                return (1, NONTRUMP_TOP[rank]) if suit == lead_suit else (0, -1)
            else:
                return (1, NONTRUMP_BOTTOM[rank]) if suit == lead_suit else (0, -1)
        best_idx = max(range(4), key=lambda i: power(trick[i]))
        return players[best_idx]


# Simple manual model for CLI testing remains unchanged
class TestJassModel(JassModel):
    def __init__(self, name: str = "Player"):
        self.name = name
    def _print_card(self, card: int) -> str:
        if card == 0:
            return "-"
        suit, rank = decode_card(card)
        suit_symbols = ["♦", "♥", "♠", "♣"]
        rank_symbols = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return f"{rank_symbols[rank]}{suit_symbols[suit]}"
    def act(self, state: "JassState", mask: torch.Tensor = None) -> torch.Tensor:
        print(f"History: {[self._print_card(c) for c in state.history]}")
        print(f"Table:   {[self._print_card(c) for c in state.table]}")
        print(f"Hand:    {[self._print_card(c) for c in state.hand]}")
        print(f"Gewisen: {[self._print_card(c) for c in state.gewisen]}")
        print(f"Trump:   {TRUMP_MODES[state.trump]}")
        if mask is not None:
            legal_slots = [i for i, ok in enumerate(mask.tolist()) if ok]
            print(f"Legal slots: {legal_slots}")
        idx = int(input("Choose a slot to play (0-8): "))
        if not (0 <= idx <= 8):
            raise ValueError("Slot must be between 0 and 8.")
        if state.hand[idx] == 0:
            raise ValueError(f"Slot {idx} is empty.")
        if mask is not None and not bool(mask[idx].item()):
            raise ValueError(f"Slot {idx} is illegal under current rules.")
        return torch.tensor(idx, dtype=torch.long)
    def choose_trump(self, state: "TrumpChoiceState") -> int:
        print(f"Trump choice — Hand: {[self._print_card(c) for c in state.hand]}")
        print(f"Must choose: {state.must_choose} (0=no, 1=yes)")
        print("Options: 0=Pass, 1=Diamonds, 2=Hearts, 3=Spades, 4=Clubs, 5=TopDown, 6=BottomUp")
        choice = int(input("Choose trump (0-6): "))
        if not (0 <= choice <= 6):
            raise ValueError("Trump must be between 0 and 6.")
        if state.must_choose == 1 and choice == 0:
            raise ValueError("Must choose a trump, cannot pass.")
        return choice


if __name__ == "__main__":
    #model = TestJassModel(name="Test Player")
    JASS_MODEL_PATH = "../models/play/JassPlay_512_256_128_dnn.pth"
    jass_model = ModelDNN(name="jass", input_size=72, embedding_size=13, hidden_size=[512, 256, 128], output_size=9)
    jass_model.load_state_dict(torch.load(JASS_MODEL_PATH, map_location='cpu'))
    jass_model.eval()
    TRUMP_MODEL_PATH = "../models/trump/JassTrump_128_64_dnn.pth"
    trump_model = ModelDNN(name="trump", input_size=10, embedding_size=13, hidden_size=[128, 64], output_size=7)
    trump_model.load_state_dict(torch.load(TRUMP_MODEL_PATH, map_location='cpu'))
    trump_model.eval()

    model = JassModel(model_jass=jass_model, model_trump=trump_model, device='cpu')

    env = JassEnv()
    S, A = env.play_game(model=model, B=1)
    print("Game finished.")
    print(f"States shape: {S.shape}")
    print(f"Actions shape: {A.shape}")
