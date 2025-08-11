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
        self.model_jass = model_jass
        self.model_trump = model_trump
        self.device = device

    def act(self, state: "JassState", mask: tensor) -> int:
        x = state.make_tensor().to(self.device)
        logits = self.model_jass(x)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        return int(F.gumbel_softmax(masked_logits, tau=1.0, hard=True).argmax(dim=-1).item())

    def choose_trump(self, state: "TrumpChoiceState") -> int:
        with torch.no_grad():
            x = state.make_tensor().to(self.device)
            logits = self.model_trump(x)  # [1,7]
            mask = torch.ones(7, dtype=torch.float, device=self.device)
            if state.must_choose == 1:
                mask[0] = 0
            masked_logits = logits.masked_fill(mask == 0, float('-inf'))
            return int(F.gumbel_softmax(masked_logits, tau=1.0, hard=True).argmax(dim=-1).item())


@dataclass
class Transition:
    obs: torch.Tensor         # [72]
    action_idx: int           # 0..8 (slot in hand)  or 0..6 during trump-choice
    action_card: int          # 1..36 (or 0 for trump-choice)
    next_obs: torch.Tensor    # [72]
    done: bool
    info: dict                # e.g., {"phase": "play"|"trump", "player": i, "legal_mask": mask(9), "trick_i": k}


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


class JassEnv:
    def __init__(self, model: JassModel):
        self.model = model
        self.trump_caller = 0

    # ---------- Public helpers for trainer ----------
    @staticmethod
    def hand_from_state(state_b72: torch.Tensor) -> torch.Tensor:
        """Extract hand slice from packed state. Layout: 32 hist | 3 table | 9 hand | 27 gewisen | 1 trump."""
        return state_b72[:, 35:44].long()  # [B,9]

    @staticmethod
    def slot_to_card_idx0(state_b72: torch.Tensor, slot_idx_b1: torch.Tensor) -> torch.Tensor:
        """Map slot (0..8) to global card index (0..35) using hand slice."""
        hand = JassEnv.hand_from_state(state_b72)  # [B,9] with 1..36 or 0
        slot = slot_idx_b1.long().clamp(min=0, max=8)
        cards = hand.gather(1, slot)  # [B,1]
        # fallback if empty slot (shouldn't happen for legal plays)
        first_nonzero = (hand != 0).long().argmax(dim=1, keepdim=True)
        cards = torch.where(cards == 0, hand.gather(1, first_nonzero), cards)
        return cards.add(-1).clamp(min=0)

    # ---------- Core gameplay ----------
    def play_game(self, *, return_masks: bool = False, include_trump_steps: bool = False, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        states: List[torch.Tensor] = []
        actions_slots: List[List[int]] = []
        actions_cards: List[List[int]] = []
        masks_all: List[torch.Tensor] = []

        deck = list(range(1, NUM_CARDS + 1))
        random.shuffle(deck)
        hands = [sorted(deck[i * 9:(i + 1) * 9]) for i in range(4)]

        # --- Trump choice ---
        with torch.no_grad():
            tc = TrumpChoiceState(hand=hands[self.trump_caller], must_choose=0)
            trump_choice = self.model.choose_trump(tc)  # int 0..6
            if trump_choice == 0:
                partner = (self.trump_caller + 2) % 4
                tc2 = TrumpChoiceState(hand=hands[partner], must_choose=1)
                trump_choice = self.model.choose_trump(tc2)  # int 1..6

        if trump_choice == 0:
            raise ValueError("No trump chosen, game cannot proceed.")

        trump = trump_choice
        self.trump_caller = (self.trump_caller + 1) % 4

        history: List[int] = []
        table: List[int] = []

        initial_hands = [h[:] for h in hands]
        gewisen_all: List[List[int]] = [[0] * 27 for _ in range(4)]
        has_revealed_weisen = [False] * 4

        current_leader = self.trump_caller
        for trick_i in range(9):
            trick: List[int] = []
            players_in_trick: List[int] = []

            for turn in range(4):
                player_i = (current_leader + turn) % 4

                state = JassState(
                    history=history + [0] * (32 - len(history)),
                    table=table + [0] * (3 - len(table)),
                    hand=hands[player_i] + [0] * (9 - len(hands[player_i])),
                    gewisen=gewisen_all[player_i][:],
                    trump=trump,
                )

                s_t = state.make_tensor()
                legal_mask = self._legal_card_mask(hands[player_i], table, trump)
                with torch.no_grad():
                    slot_idx = self.model.act(state, legal_mask)
                chosen_card = hands[player_i].pop(slot_idx)

                # log
                states.append(s_t)
                actions_slots.append([slot_idx])
                actions_cards.append([chosen_card])
                if return_masks:
                    masks_all.append(legal_mask)

                # play
                table.append(chosen_card)
                trick.append(chosen_card)
                players_in_trick.append(player_i)

                # reveal weise after first play by player
                if not has_revealed_weisen[player_i]:
                    has_revealed_weisen[player_i] = True
                    weise_cards = self._detect_weisen(initial_hands[player_i], trump=trump)
                    for viewer in range(4):
                        if viewer == player_i:
                            continue
                        relative_pos = (player_i - viewer - 1) % 4
                        if relative_pos >= 3:
                            continue
                        base_idx = relative_pos * 9
                        seg = weise_cards[:9] + [0] * (9 - min(9, len(weise_cards)))
                        gewisen_all[viewer][base_idx:base_idx + 9] = seg

            winner = self._determine_trick_winner(trick, players_in_trick, trump)
            history.extend(trick)
            table.clear()
            current_leader = winner

        # pack outputs
        S = torch.cat(states, dim=0).to(torch.long)            # [T,72]
        A_slot = torch.tensor(actions_slots, dtype=torch.long) # [T,1]
        A_card = torch.tensor(actions_cards, dtype=torch.long) # [T,1]

        if return_masks:
            M = torch.stack([m.to(torch.uint8) for m in masks_all], dim=0) if masks_all else torch.empty((0, 9), dtype=torch.uint8)
            return S, A_slot, A_card, M
        return S, A_slot, A_card

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
    model = TestJassModel(name="Test Player")
    env = JassEnv(model)
    S, A_slot, A_card = env.play_game()
    print("Game finished.")
    print(f"States shape: {S.shape}")
    print(f"Slot actions shape: {A_slot.shape}, Card actions shape: {A_card.shape}")
