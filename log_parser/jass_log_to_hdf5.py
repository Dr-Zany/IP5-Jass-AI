# -*- coding: utf-8 -*-
"""
Parses Jass game logs and stores state-action pairs efficiently in an HDF5 file.
Uses standard logging for output.
Increased PLAYER_ID_MAX_LEN to handle longer IDs.
"""

import re
import json
import h5py
import numpy as np
from bitarray import bitarray
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import argparse
import logging # Import the logging module
import os
import traceback # Needed for logging exceptions

# --- Logging Configuration ---
# Configure logging to output messages to the console
# You can adjust the level (e.g., logging.DEBUG) and format as needed
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler("jass_log_parser.log"), # Log to a file
                        logging.StreamHandler() # Also log to console
                    ])


# --- Constants ---
NUM_CARDS_HAND = 9
NUM_CARDS_TABLE = 3 # Max cards on table in one trick
NUM_CARDS_HISTORY = 32 # Max cards played before current trick in a round (8 tricks * 4 cards)
NUM_CARDS_TOTAL_ROUND = 36
CARD_BITS = 13
TRUMP_BITS = 6
STATE_BITS = (NUM_CARDS_HISTORY * CARD_BITS) + \
             (NUM_CARDS_TABLE * CARD_BITS) + \
             (NUM_CARDS_HAND * CARD_BITS) + \
             TRUMP_BITS # 32*13 + 4*13 + 9*13 + 6 = 416 + 52 + 117 + 6 = 591 bits
ACTION_BITS = CARD_BITS # 13 bits for the card played
# Increased max length for player ID string to avoid truncation

# --- Class Definitions ---

class Card:
    """
    Represents a Jass card using a 13-bit bitarray.

    Bits 0-8: Value (6, 7, 8, 9, 10, Jack, Queen, King, Ace)
    Bits 9-12: Suit (Hearts, Diamonds, Clubs, Spades)
    """
    def __init__(self, card_num: Optional[int] = None):
        """
        Initializes a Card object.

        Args:
            card_num: An integer from 0 to 35 representing the card,
                      or None to create an empty/invalid card (all bits 0).
                      Mapping:
                      0-8: Hearts (6-Ace)
                      9-17: Diamonds (6-Ace)
                      18-26: Clubs (6-Ace)
                      27-35: Spades (6-Ace)
        """
        self.card = bitarray(CARD_BITS)
        self.card.setall(0) # Initialize as an empty card
        if card_num is not None and 0 <= card_num < NUM_CARDS_TOTAL_ROUND:
            value_index = card_num % NUM_CARDS_HAND # 0-8
            suit_index = (card_num // NUM_CARDS_HAND) + NUM_CARDS_HAND # 9-12
            # Check indices are valid before setting bits
            if 0 <= value_index < NUM_CARDS_HAND and NUM_CARDS_HAND <= suit_index < CARD_BITS:
                self.card[value_index] = 1
                self.card[suit_index] = 1
            else:
                # This case should ideally not happen with valid card_num input
                logging.warning(f"Invalid card indices derived from card_num {card_num}.")


    def to_array(self) -> np.ndarray:
        """Converts the card's bitarray to a NumPy boolean array."""
        return np.array(self.card.tolist(), dtype=bool)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Card':
        """Creates a Card object from a NumPy boolean array."""
        instance = cls()
        if arr.size == CARD_BITS:
             instance.card = bitarray(arr.tolist())
        else:
             logging.warning(f"Array size mismatch in Card.from_array. Expected {CARD_BITS}, got {arr.size}.")
             # Keep the default empty card
        return instance

    def is_empty(self) -> bool:
        """Checks if the card is an empty card (all bits zero)."""
        return not self.card.any()

    def __repr__(self) -> str:
        return f"Card({self.card.to01()})"

class Trump:
    """
    Represents the trump suit using a 6-bit bitarray.

    Bit 0: Hearts
    Bit 1: Diamonds
    Bit 2: Clubs
    Bit 3: Spades
    Bit 4: Bottom-up (Undeufe)
    Bit 5: Top-down (Obenabe)
    """
    def __init__(self, trump_num: Optional[int] = None):
        """
        Initializes the Trump object.

        Args:
            trump_num: An integer 0-5 representing the trump suit,
                       or None for no trump initially.
                       Handles trump 7 (switch) separately.
        """
        self.value = bitarray(TRUMP_BITS)
        self.value.setall(0)
        if trump_num is not None and 0 <= trump_num < TRUMP_BITS:
            self.value[trump_num] = 1

    def set_trump(self, trump_num: int) -> None:
        """Sets the trump suit. Returns True if it was a 'switch'."""
        self.value.setall(0)
        if trump_num == -1: # No trump selected yet / Pass
            pass # Pass to m8
        elif 0 <= trump_num < TRUMP_BITS:
            self.value[trump_num] = 1
        else:
            logging.warning(f"Invalid trump_num {trump_num} passed to set_trump.")

    def to_array(self) -> np.ndarray:
        """Converts the trump bitarray to a NumPy boolean array."""
        return np.array(self.value.tolist(), dtype=bool)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Trump':
        """Creates a Trump object from a NumPy boolean array."""
        instance = cls()
        if arr.size == TRUMP_BITS:
            instance.value = bitarray(arr.tolist())
        else:
            logging.warning(f"Array size mismatch in Trump.from_array. Expected {TRUMP_BITS}, got {arr.size}.")
        return instance

    def __repr__(self) -> str:
        return f"Trump({self.value.to01()})"

class State:
    """
    Represents the game state from the perspective of one player *before* they play.

    Includes fixed-size representations of:
    - History: Cards played in completed tricks this round (padded to 32).
    - Table: Cards played in the current trick *before* this player (padded to 4).
    - Hand: Cards currently in the player's hand (padded to 9).
    - Trump: The current trump suit.
    - Player ID: The ID of the player whose perspective this is.
    """
    def __init__(self,
                 cards_player: List[Card],
                 cards_table: List[Card],
                 cards_history: List[Card],
                 trump: Trump):
        """
        Initializes the State object, ensuring all card lists are padded
        to their fixed maximum sizes with empty Card objects.
        """
        # --- Padding Logic ---
        self.cards_player: List[Card] = (cards_player + [Card()] * NUM_CARDS_HAND)[:NUM_CARDS_HAND]
        self.cards_table: List[Card] = (cards_table + [Card()] * NUM_CARDS_TABLE)[:NUM_CARDS_TABLE]
        self.cards_history: List[Card] = (cards_history + [Card()] * NUM_CARDS_HISTORY)[:NUM_CARDS_HISTORY]
        # --- End Padding ---

        self.trump: Trump = trump

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts the state into NumPy arrays suitable for saving.

        Returns:
            A tuple containing:
            - state_bits (np.ndarray): Boolean array of size STATE_BITS, concatenating
                                       history, table, hand, and trump bits.
            - player_id_arr (np.ndarray): NumPy array holding the player ID as bytes.
        """
        state_bits = bitarray()
        # Extend with history, table, hand, and trump bits in order
        for card in self.cards_history: # Always 32 cards (padded)
            state_bits.extend(card.card)
        for card in self.cards_table:   # Always 3 cards (padded)
            state_bits.extend(card.card)
        for card in self.cards_player:  # Always 9 cards (padded)
            state_bits.extend(card.card)
        state_bits.extend(self.trump.value) # 6 bits

        # Verification: Check if the final length matches STATE_BITS
        if len(state_bits) != STATE_BITS:
             # This should not happen if constants and padding are correct.
             logging.error(f"State bit length mismatch! Expected {STATE_BITS}, got {len(state_bits)}. Check constants and padding.")
             # Handle error appropriately, maybe raise exception or return invalid data
             # For now, pad/truncate defensively, but this indicates a bug.
             state_bits.extend([0] * (STATE_BITS - len(state_bits)))
             state_bits = state_bits[:STATE_BITS]


        state_array = np.array(state_bits.tolist(), dtype=bool)
        return state_array

    def __repr__(self) -> str:
        # Count non-empty cards for a more informative representation
        hand_count = sum(1 for card in self.cards_player if not card.is_empty())
        table_count = sum(1 for card in self.cards_table if not card.is_empty())
        history_count = sum(1 for card in self.cards_history if not card.is_empty())
        return (f"State(Player: {self.player_id}, Hand: {hand_count}/{NUM_CARDS_HAND}, "
                f"Table: {table_count}/{NUM_CARDS_TABLE}, History: {history_count}/{NUM_CARDS_HISTORY}, Trump: {self.trump})")


class Action:
    """
    Represents the action taken by a player.

    Includes:
    - Card Played: The card played by the player.
    - Time To Play: Time elapsed in seconds since the last action.
    """
    def __init__(self, card_played: Card, time_to_play: float):
        self.card_played: Card = card_played
        self.time_to_play: float = max(0.0, time_to_play) # Ensure time is not negative

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts the action into NumPy arrays suitable for saving.

        Returns:
            A tuple containing:
            - card_bits (np.ndarray): Boolean array of size CARD_BITS for the card.
            - time_arr (np.ndarray): Float array containing the time to play.
        """
        card_array = self.card_played.to_array()
        time_array = np.array([self.time_to_play], dtype=float)
        return card_array, time_array

    def __repr__(self) -> str:
        return f"Action(Card: {self.card_played}, Time: {self.time_to_play:.2f}s)"

# --- Regex Definitions ---
r_playerInfo = re.compile(r'\"usernickname\":\s*\"([\w\d]+)\".*?\"eq\":\s*(\d+).*?\"iq\":\s*(\d+).*?\"niceness\":\s*(\d+).*?\"honness\":\s*(\d+).*?\"winness\":\s*(\d+).*?\"playedgames\":\s*(\d+).*?\"profival\":\s*(\d+)', re.IGNORECASE)
r_newRound = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"action\":{\"newRound\": \"Runde (\d+)\"', re.IGNORECASE)
r_cardsDealt = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"action\":{\"doDeal\":\s*\d+,\"player\":\s*\d+,\s*\"usernickname\":\s*\"([\w\d]+)\",\"cardset\":\s*\[\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\"\]', re.IGNORECASE)
r_setTrump = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":{\"submitTrump\":\s*(-?\d+)', re.IGNORECASE)
r_playCard = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":{\"submitsCard\":\s*\"(\d+)\"', re.IGNORECASE)
r_gameEnd = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":\s*\"gameFinished\"}', re.IGNORECASE)



# --- Parsing Function ---

def parse_file(filename: str, hdf5_file: h5py.File) -> None:
    """
    Parses a Jass log file and saves state-action pairs to an HDF5 file.

    Args:
        filename: Path to the log file.
        hdf5_file: An opened h5py.File object for writing.
    """
    # Extract a base name for the HDF5 group
    base_filename = os.path.basename(filename) # Use os.path.basename for safety
    game_group_name = os.path.splitext(base_filename)[0] # Use os.path.splitext

    # Reset game-specific state for each file
    player_infos: List[Dict[str, Any]] = []
    player_ids_in_game: List[str] = []
    found_all_players = False # Flag to track if player info is complete for this game

    if game_group_name in hdf5_file:
        logging.warning(f"Group '{game_group_name}' already exists in HDF5 file. Overwriting.")
        del hdf5_file[game_group_name]
    group = hdf5_file.create_group(game_group_name)
    logging.debug(f"Created HDF5 group: /{game_group_name}")

    # --- HDF5 Dataset Initialization ---
    # Create resizable datasets with chunking for better I/O performance
    dset_state_bits = group.create_dataset("state_bits", (0, STATE_BITS), maxshape=(None, STATE_BITS), dtype='bool', chunks=(128, STATE_BITS), compression="gzip")
    dset_state_player = group.create_dataset("state_player_id", (0,), maxshape=(None,), dtype=f'uint8', chunks=(128,), compression="gzip")

    # Action data
    dset_action_card = group.create_dataset("action_card_bits", (0, ACTION_BITS), maxshape=(None, ACTION_BITS), dtype='bool', chunks=(128, ACTION_BITS), compression="gzip")
    dset_action_time = group.create_dataset("action_time", (0,), maxshape=(None,), dtype='float', chunks=(128,), compression="gzip")

    # --- Game State Variables (reset per file) ---
    time_last_action: Optional[datetime] = None
    hands: Dict[str, List[Card]] = {} # Current cards in each player's hand
    history: List[Card] = [] # Cards from completed tricks in the current round
    table: List[Tuple[str, Card]] = [] # Cards on table in the current trick [(player_id, card), ...]
    current_trick_cards: List[Card] = [] # Just the cards on table for state creation
    round_number: int = 0
    trick_counter: int = 0 # Counts cards played in the current trick (0-3)
    round_card_counter: int = 0 # Counts total cards played in the round (0-35)
    trump: Trump = Trump() # Current trump suit

    logging.debug(f"Processing file: {filename}")
    line_count = 0
    saved_states = 0

    discarded = False

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_count = i + 1
                line = line.strip()
                if not line: continue # Skip empty lines

                # --- Event Matching ---

                # Match Player Info (New Game)
                m_player = r_playerInfo.search(line)
                if m_player and not found_all_players:
                    p_info = {
                        "usernickname": m_player.group(1),
                        "eq": int(m_player.group(2)), "iq": int(m_player.group(3)),
                        "niceness": int(m_player.group(4)), "honness": int(m_player.group(5)),
                        "winness": int(m_player.group(6)), "playedgames": int(m_player.group(7)),
                        "profival": int(m_player.group(8))
                    }

                    player_infos.append(p_info)
                    player_ids_in_game.append(p_info["usernickname"])
                    logging.debug(f"Found player {p_info['usernickname']}")

                    if len(player_infos) == 4:
                        # Save player info once 4 players are found
                        group.attrs["players"] = json.dumps(player_infos)
                        logging.debug(f"Found 4 players for game {game_group_name}: {player_ids_in_game}")
                        found_all_players = True # Stop looking for player info
                        # Initialize hands structure now that we know the players
                        hands = {pid: [] for pid in player_ids_in_game}
                    continue # Move to next line after processing player info

                # Match New Round
                m_round = r_newRound.match(line)
                if m_round:
                    if not found_all_players:
                        logging.warning(f"Line {line_count}: New round started before finding all 4 players in {filename}. Player list might be incomplete.")
                        # Attempt to continue, but state might be unreliable
                    round_num_str = m_round.group(2)
                    logging.debug(f"--- New Round {round_num_str} ---")
                    # Reset round-specific state
                    try:
                        time_last_action = datetime.strptime(m_round.group(1), "%Y-%m-%d %H:%M:%S%z")
                        round_number = int(round_num_str)
                    except ValueError as e:
                        logging.error(f"Line {line_count}: Could not parse time or round number ({e}). Skipping round reset.")
                        continue # Skip if critical info is missing

                    # Reset round state variables
                    hands = {pid: [] for pid in player_ids_in_game} # Ensure hands are reset based on known players
                    history = []
                    table = []
                    current_trick_cards = []
                    trick_counter = 0
                    round_card_counter = 0
                    trump = Trump() # Reset trump
                    continue

                # Match Cards Dealt
                m_deal = r_cardsDealt.match(line)
                if m_deal:
                    try:
                        deal_time = datetime.strptime(m_deal.group(1), "%Y-%m-%d %H:%M:%S%z")
                        player_id = m_deal.group(2)
                        # Card numbers start from group 3
                        card_nums = [int(card_str) for card_str in m_deal.groups()[2:]]
                        dealt_cards = [Card(num) for num in card_nums]

                        if player_id in hands:
                            hands[player_id] = dealt_cards
                            # Update time_last_action if this is the most recent event
                            if time_last_action is None or deal_time > time_last_action:
                                time_last_action = deal_time
                            logging.debug(f"Dealt cards to {player_id}")
                        else:
                            # This might happen if player info wasn't found before dealing
                            logging.warning(f"Line {line_count}: Player {player_id} dealt cards but not in known players {player_ids_in_game}. Hand ignored.")

                    except (ValueError, IndexError) as e:
                        logging.error(f"Line {line_count}: Error parsing card deal ({e}). Skipping.")
                        discarded = True
                        break
                    continue

                # Match Set Trump
                m_trump = r_setTrump.match(line)
                if m_trump:
                    try:
                        trump_time = datetime.strptime(m_trump.group(1), "%Y-%m-%d %H:%M:%S%z")
                        player_id = m_trump.group(2)
                        trump_val = int(m_trump.group(3))

                        if player_id not in player_ids_in_game:
                             logging.warning(f"Line {line_count}: Player {player_id} set trump but not in known players {player_ids_in_game}. Trump setting ignored.")
                             continue

                        logging.debug(f"Player {player_id} sets trump to {trump_val}")
                        trump.set_trump(trump_val)

                        if time_last_action is None or trump_time > time_last_action:
                             time_last_action = trump_time
                    except (ValueError, IndexError) as e:
                        logging.error(f"Line {line_count}: Error parsing trump setting ({e}). Skipping.")
                    continue

                # Match Play Card
                m_play = r_playCard.match(line)
                if m_play:
                    try:
                        time_new_action = datetime.strptime(m_play.group(1), "%Y-%m-%d %H:%M:%S%z")
                        player_id = m_play.group(2)
                        card_played_num = int(m_play.group(3))
                        card_played = Card(card_played_num)

                        
                        # --- Pre-computation Checks ---
                        if not found_all_players:
                            logging.error(f"Line {line_count}: Card played by {player_id} before all players identified. Skipping state save.")
                            discarded = True
                            break
                        if player_id not in hands:
                            logging.error(f"Line {line_count}: Player {player_id} played card but no hand structure found (maybe missed player info?). Skipping state save.")
                            discarded = True
                            break
                        if not hands[player_id]: # Check if hand is unexpectedly empty
                            logging.warning(f"Line {line_count}: Player {player_id} attempting to play card {card_played_num} but hand is recorded as empty. Skipping state save. {filename}")
                            discarded = True
                            break
                        if time_last_action is None:
                             logging.warning(f"Line {line_count}: Card played by {player_id} but no previous timestamp. Assuming 0 time delta. State time might be inaccurate.")
                             time_delta_seconds = 0.0
                             # Set time_last_action to current time to allow next step
                             time_last_action = time_new_action
                        else:
                             time_delta = time_new_action - time_last_action
                             time_delta_seconds = time_delta.total_seconds()


                        # --- Create State (Player's perspective BEFORE playing) ---
                    
                        current_hand = list(hands[player_id])
                        current_history = list(history)
                        current_table_cards_only = list(current_trick_cards)

                        state = State(current_hand, current_table_cards_only, current_history, trump)

                        # --- Create Action ---
                        action = Action(card_played, time_delta_seconds)

                        # --- Save State and Action to HDF5 ---
                        state_arr = state.to_arrays()
                        action_card_arr, action_time_arr = action.to_arrays()

                        # Resize datasets before appending
                        current_size = dset_state_bits.shape[0]
                        dset_state_bits.resize(current_size + 1, axis=0)
                        dset_state_player.resize(current_size + 1, axis=0)
                        dset_action_card.resize(current_size + 1, axis=0)
                        dset_action_time.resize(current_size + 1, axis=0)

                        # Append data to the last row
                        dset_state_bits[current_size, :] = state_arr
                        dset_state_player[current_size] = list(hands.keys()).index(player_id)
                        dset_action_card[current_size, :] = action_card_arr
                        dset_action_time[current_size] = action_time_arr[0]

                        saved_states += 1

                        # --- Update Game State AFTER saving ---
                        card_found_in_hand = False
                        for idx, hand_card in enumerate(hands[player_id]):
                            if hand_card.card == card_played.card:
                                del hands[player_id][idx]
                                card_found_in_hand = True
                                break
                        if not card_found_in_hand:
                            logging.warning(f"Line {line_count}: Card {card_played_num} ({card_played}) played by {player_id} was not found in their current hand!")
                            logging.warning(f"  Current hand state before play: {[c for c in current_hand]}")
                            # Attempt to continue, but state might be inconsistent

                        table.append((player_id, card_played))
                        current_trick_cards.append(card_played)

                        time_last_action = time_new_action
                        trick_counter += 1
                        round_card_counter += 1

                        # Check if trick is complete
                        if trick_counter == 4:
                            logging.debug(f"Trick {round_card_counter // 4} complete.")
                            history.extend([c for p, c in table])
                            table = []
                            current_trick_cards = []
                            trick_counter = 0

                        # Check if round is complete
                        if round_card_counter == NUM_CARDS_TOTAL_ROUND:
                             logging.debug(f"--- Round {round_number} Complete ---")
                             # State resets automatically on next 'newRound' match

                    except (ValueError, IndexError, KeyError) as e:
                         logging.error(f"Line {line_count}: Failed to process card play for player {m_play.group(2)} card {m_play.group(3)} ({e}). Skipping.")
                         logging.debug(f"  Problematic line: {line}")
                         logging.exception("Traceback:")
                    continue # Ensure we move to the next line even if errors 
                
                # Match Game End
                m_end = r_gameEnd.match(line)
                if m_end:
                    logging.debug(f"Game ended")
                    break # End of file reached, break out of loop

    except FileNotFoundError:
        logging.error(f"Log file not found at '{filename}'")
        return # Skip this file
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing file '{filename}' at line ~{line_count}: {e}")
        logging.exception("Traceback:")
        # Decide if you want to stop processing this file or continue
        return # Stop processing this file on major errors

    # Add final attributes after processing the file
    group.attrs['total_states_saved'] = saved_states
    group.attrs['log_lines_processed'] = line_count
    if not found_all_players and saved_states > 0:
         logging.warning(f"Finished processing {filename}, but player info might be incomplete. Saved {saved_states} states.")
    elif saved_states > 0:
         logging.debug(f"Finished processing {filename}. Saved {saved_states} state-action pairs to group /{game_group_name}.")
    else:
         logging.warning(f"Finished processing {filename}. No state-action pairs saved (possibly due to missing info or errors).")

    if discarded:
        logging.warning(f"Discarded file {filename} due to critical errors. No data saved.")
        del group # Remove the group if no valid data was saved


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Jass game log files and save state-action pairs to HDF5.")
    parser.add_argument("-i","--input-dir", required=True, help="Directory containing the .game log files.")
    parser.add_argument("-o","--output-file", required=True, help="Path to the output HDF5 file.")
    args = parser.parse_args()

    # Open HDF5 file for appending ('a' mode)
    try:
        with h5py.File(args.output_file, 'a') as hdf5_file:
            logging.info(f"Opened HDF5 file: {args.output_file} in append mode.")
            files_processed_count = 0
            if not os.path.isdir(args.input_dir):
                logging.error(f"Input directory not found or is not a directory: {args.input_dir}")
            else:
                all_files = os.listdir(args.input_dir)
                # Filter for files ending with .game, case-insensitive
                game_files = [f for f in all_files if f.lower().endswith(".game") and os.path.isfile(os.path.join(args.input_dir, f))]
                total_game_files = len(game_files)
                logging.info(f"Found {total_game_files} .game files in {args.input_dir}")

                for i, filename in enumerate(game_files):
                    full_path = os.path.join(args.input_dir, filename)
                    # parse_file handles its own file-level errors
                    parse_file(full_path, hdf5_file)
                    files_processed_count += 1

                    if (i + 1) % 10 == 0: # Log progress every 10 files
                        logging.info(f"Processed {i + 1}/{total_game_files} files from {args.input_dir}...")

            logging.info(f"Finished processing all potential .game files. Processed {files_processed_count} files.")
            logging.info(f"Final HDF5 file saved to: {args.output_file}")

    except Exception as e:
        # Catch errors related to opening/writing the HDF5 file itself
        logging.critical(f"Failed to open or write to HDF5 file {args.output_file}: {e}")
        logging.exception("Traceback:")
