import argparse
from datetime import datetime
import h5py
import re
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import tqdm

NUM_CARDS_HAND = 9
NUM_CARDS_TABLE = 3
NUM_CARDS_HISTORY = 32
NUM_CARDS_SHOWN = 27 # 9 cards shown for each player 
NUM_TRUMPS = 1

NUM_STATE = NUM_CARDS_HAND + NUM_CARDS_TABLE + NUM_CARDS_HISTORY + NUM_CARDS_SHOWN + NUM_TRUMPS # 1 for trump, 9 for each player, 32 for history, 27 for shown cards

# --- Logging Configuration ---

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler("jass_log_parser.log"), # Log to a file
                    ])

# --- Regex Definitions ---
r_playerInfo = re.compile(r'\"usernickname\":\s*\"([\w\d]+)\".*?\"eq\":\s*(\d+).*?\"iq\":\s*(\d+).*?\"niceness\":\s*(\d+).*?\"honness\":\s*(\d+).*?\"winness\":\s*(\d+).*?\"playedgames\":\s*(\d+).*?\"profival\":\s*(\d+)', re.IGNORECASE)
r_newRound = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"action\":{\"newRound\": \"Runde (\d+)\"', re.IGNORECASE)
r_cardsDealt = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"action\":{\"doDeal\":\s*(\d+),\"player\":\s*\d+,\s*\"usernickname\":\s*\"([\w\d]+)\",\"cardset\":\s*\[\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\"\]', re.IGNORECASE)
r_setTrump = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":{\"submitTrump\":\s*(-?\d+)', re.IGNORECASE)
r_playCard = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":{\"submitsCard\":\s*\"(\d+)\"', re.IGNORECASE)
r_cardShow = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":{\"doRequestSuits\":\s*\[".*?;((?:\d+,?)+)"',re.IGNORECASE)
r_gameEnd = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":\s*\"gameFinished\"}', re.IGNORECASE)
r_point = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":{.*?\"gameTotal\":\s\[Rundentotal,(\d+),\d+\]', re.IGNORECASE)

# --- State Functions ---
def create_state(hand: List[int], table: List[int], history: List[int], shown: List[List[int]], trump: int) -> List[int]:
    """
    Create a state representation for the Jass game.

    Args:
        hand (List[int]): Cards in hand.
        table (List[int]): Cards on the table.
        history (List[int]): Cards in history.
        shown (List[List[int]]): Cards shown by each player.
        trump (int): Trump suit.

    Returns:
        List[int]: State representation as a list of integers.
    """
    state = []
    state += history + [0] * (NUM_CARDS_HISTORY - len(history)) # Fill with zeros if history is shorter than expected
    state += table + [0] * (NUM_CARDS_TABLE - len(table)) # Fill with zeros if table is shorter than expected
    state += hand + [0] * (NUM_CARDS_HAND - len(hand)) # Fill with zeros if hand is shorter than expected

    if len(shown) != 3:
        logging.error(f"Shown cards length mismatch: expected 3, got {len(shown)}")
        raise ValueError(f"Shown cards length mismatch: expected 3, got {len(shown)}")

    for shown_cards in shown:
        state += shown_cards + [0] * (9 - len(shown_cards))
    state += [trump] # Add trump value
    if len(state) != NUM_STATE:
        logging.error(f"State length mismatch: expected {NUM_STATE}, got {len(state)}")
        raise ValueError(f"State length mismatch: expected {NUM_STATE}, got {len(state)}")

    return state

def create_state_with_points(hand: List[int], table: List[int], history: List[int], shown: List[List[int]], trump: int, points: int, other_points: int) -> List[int]:
    """
    Create a state representation for the Jass game with points.

    Args:
        hand (List[int]): Cards in hand.
        table (List[int]): Cards on the table.
        history (List[int]): Cards in history.
        shown (List[List[int]]): Cards shown by each player.
        trump (int): Trump suit.
        points (int): Points for the player.

    Returns:
        List[int]: State representation as a list of integers.
    """
    state = create_state(hand, table, history, shown, trump)
    state.append(points)  # Append points to the state
    state.append(other_points)  # Append other player's points
    return state

def create_action(card: int, hand: list[int]) -> int:
    """
    Create an action representation for the Jass game.

    Args:
        card (int): Card played.
        hand (list[int]): Cards in hand.

    Returns:
        int: Action representation as an integer.
    """
    return hand.index(card) if card in hand else -1


# --- Parser Functions ---
def parse_file(file_path: str, hdf5_play: h5py.File, hdf5_trump: h5py.File, hdf5_time: h5py.File, hdf5_play_with_points: h5py.File, hdf5_trump_with_points: h5py.File, hdf5_play_cheated: h5py.File) -> int:
    """
    Parse the Jass log file and store the data in an HDF5 file.

    Args:
        file_path (str): Path to the Jass log file.
        hdf5_file (h5py.file): HDF5 file object to store the parsed data.
    """

    file_name = os.path.basename(file_path)
    game_group_name = os.path.splitext(file_name)[0] 

    player_ids_in_game: List[str] = []
    found_all_players = False # Flag to track if player info is complete for this game

    group_play_with_points = hdf5_play_with_points.create_group(game_group_name)
    group_trump_with_points = hdf5_trump_with_points.create_group(game_group_name)
    group_play_cheated = hdf5_play_cheated.create_group(game_group_name)
    group_play = hdf5_play.create_group(game_group_name)
    group_trump = hdf5_trump.create_group(game_group_name)
    group_time = hdf5_time.create_group(game_group_name)
    logging.debug(f"Created HDF5 group: /{game_group_name}")

    dset_state_play = group_play.create_dataset("state", (0, NUM_STATE), maxshape=(None, NUM_STATE), chunks=(1,NUM_STATE), dtype='uint8', compression="gzip")
    dset_state_play_with_points = group_play_with_points.create_dataset("state", (0, NUM_STATE + 2), maxshape=(None, NUM_STATE + 2), chunks=(1,NUM_STATE + 2), dtype='uint32', compression="gzip")
    dset_state_play_cheated = group_play_cheated.create_dataset("state", (0, NUM_STATE), maxshape=(None, NUM_STATE), chunks=(1,NUM_STATE), dtype='uint8', compression="gzip")
    dset_state_trump_with_points = group_trump_with_points.create_dataset("state", (0, 10 + 2), maxshape=(None, 10 + 2), chunks=True, dtype='uint32', compression="gzip")
    dset_state_trump = group_trump.create_dataset("state", (0, 10), maxshape=(None, 10), chunks=True, dtype='uint8', compression="gzip")
    dset_state_time = group_time.create_dataset("state", (0, NUM_STATE), maxshape=(None, NUM_STATE), chunks=(1,NUM_STATE), dtype='uint8', compression="gzip")

    dset_player_info_play = group_play.create_dataset("player_info", (0, 1), maxshape=(None, 1), chunks=True, dtype=h5py.string_dtype(length=32), compression="gzip")
    dset_player_info_trump = group_trump.create_dataset("player_info", (0, 1), maxshape=(None, 1), chunks=True, dtype=h5py.string_dtype(length=32), compression="gzip")
    dset_player_info_time = group_time.create_dataset("player_info", (0, 1), maxshape=(None, 1), chunks=True, dtype=h5py.string_dtype(length=32), compression="gzip")

    dset_action_play = group_play.create_dataset("action", (0, 1), maxshape=(None, 1), chunks=True, dtype='uint8', compression="gzip")
    dset_action_play_with_points = group_play_with_points.create_dataset("action", (0, 1), maxshape=(None, 1), chunks=True, dtype='uint8', compression="gzip")
    dset_action_play_cheated = group_play_cheated.create_dataset("action", (0, 1), maxshape=(None, 1), chunks=True, dtype='uint8', compression="gzip")
    dset_action_trump_with_points = group_trump_with_points.create_dataset("action", (0, 1), maxshape=(None, 1), chunks=True, dtype='uint8', compression="gzip")
    dset_action_trump = group_trump.create_dataset("action", (0, 1), maxshape=(None, 1), chunks=True, dtype='uint8', compression="gzip")
    dset_action_time = group_time.create_dataset("action", (0, 1), maxshape=(None, 1), chunks=True, dtype='float', compression="gzip")

    time_last_action: Optional[datetime] = None
    player_order: Dict[str, int] = {} # Maps player IDs to their order in the game
    player_points: Dict[str, int] = {} # Points for each player
    hands: Dict[str, List[int]] = {} # Current cards in each player's hand
    shown: List[List[int]] = {} # Cards shown by each player (so called "Weiss")
    history: List[int] = [] # Cards from completed tricks in the current round
    table: List[Tuple[str, int]] = [] # Cards on table in the current trick [(player_id, card), ...]
    current_trick_cards: List[int] = [] # Just the cards on table for state creation
    round_number: int = 0
    trick_counter: int = 0 # Counts cards played in the current trick (0-3)
    round_card_counter: int = 0 # Counts total cards played in the round (0-35)
    trump: int = 0 # Current trump suit
    trump_pushed: bool = False # Flag to track if trump has been set

    logging.debug(f"Processing file: {filename}")
    line_count = 0
    saved_states_play = 0
    saved_states_trump = 0
    saved_states_time = 0

    discarded = False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_count = i + 1
                line = line.strip()
                if not line: continue # Skip empty lines

                # --- Event Matching ---

                # Match Player Info (New Game)
                m_player = r_playerInfo.search(line)
                if m_player and not found_all_players:
                    player_ids_in_game.append(m_player.group(1))
                    niceness = int(m_player.group(4))
                    logging.debug(f"Found player {m_player.group(1)}")

                    if niceness <= 0:
                        logging.warning(f"Line {line_count}: Player {m_player.group(1)} is not nice. skipping this game.")
                        discarded = True
                        break

                    if len(player_ids_in_game) == 4:
                        # Save player info once 4 players are found
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
                    trump = 0
                    player_order = {}
                    shown = [[] for _ in range(4)]
                    trump_pushed = False # Reset trump pushed flag
                    continue

                # Match Cards Dealt
                m_deal = r_cardsDealt.match(line)
                if m_deal:
                    try:
                        deal_time = datetime.strptime(m_deal.group(1), "%Y-%m-%d %H:%M:%S%z")
                        player_id = m_deal.group(3)
                        player_order[player_id] = 3 - int(m_deal.group(2)) # Player ID and their order in the game
                        # Card numbers start from group 3
                        card_nums = [int(card_str) + 1 for card_str in m_deal.groups()[3:]]

                        if player_id in hands:
                            hands[player_id] = card_nums
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

                # Match Card Show
                m_show = r_cardShow.match(line)
                if m_show:
                    try:
                        player_id = m_show.group(2)
                        card_nums = [int(card_str) + 1 for card_str in m_show.group(3).split(",")]

                        if player_id not in player_ids_in_game:
                             logging.warning(f"Line {line_count}: Player {player_id} showed cards but not in known players {player_ids_in_game}. Show ignored.")
                             continue

                        shown[player_order[player_id]] = card_nums # Store shown cards in the order of player IDs
                        logging.debug(f"Player {player_id} showed cards: {shown[-1]}")

                    except (ValueError, IndexError) as e:
                        logging.error(f"Line {line_count}: Error parsing card show ({e}). Skipping.")
                    continue


                # Match Set Trump
                m_trump = r_setTrump.match(line)
                if m_trump:
                    try:
                        trump_time = datetime.strptime(m_trump.group(1), "%Y-%m-%d %H:%M:%S%z")
                        player_id = m_trump.group(2)
                        trump_val = int(m_trump.group(3)) + 1

                        if player_id not in hands:
                            logging.warning(f"Line {line_count}: Player {player_id} set trump but no hand structure found (maybe missed player info?). Skipping.")
                            discarded = True
                            break

                        if player_id not in player_ids_in_game:
                            logging.warning(f"Line {line_count}: Player {player_id} set trump but not in known players {player_ids_in_game}. Trump setting ignored.")
                            discarded = True
                            break

                        if trump_pushed and not trump_val in range(1, 7):
                            logging.warning(f"Line {line_count}: Player {player_id} set trump to {trump_val} but trump has already been pushed. Ignoring.")
                            discarded = True
                            break

                        current_size_trump = dset_state_trump.shape[0]
                        dset_state_trump.resize(current_size_trump + 1, axis=0)
                        dset_state_trump_with_points.resize(current_size_trump + 1, axis=0)
                        dset_player_info_trump.resize(current_size_trump + 1, axis=0)
                        dset_action_trump.resize(current_size_trump + 1, axis=0)

                        current_size_time = dset_state_time.shape[0]
                        dset_state_time.resize(current_size_time + 1, axis=0)
                        dset_player_info_time.resize(current_size_time + 1, axis=0)
                        dset_action_time.resize(current_size_time + 1, axis=0)

                        dset_state_trump[current_size_trump:] = hands[player_id] + [int(trump_pushed)]
                        previous_player = [uuid for uuid, value in player_order.items() if value == (player_order[player_id] + 1) % 4][0]
                        dset_state_trump_with_points[current_size_trump,:] = hands[player_id] + [int(trump_pushed), int(player_points.get(player_id, 0)), int(player_points.get(previous_player,0))] # Add points if available
                        dset_action_trump_with_points[current_size_trump:] = [trump_val]
                        dset_player_info_trump[current_size_trump] = player_id
                        dset_action_trump[current_size_trump:] = [trump_val]
                        saved_states_trump += 1

                        state_time = create_state(hands[player_id], [], [], [[],[],[]], 0) # 0 to indicate time is for trumping
                        dset_state_time[current_size_time:] = state_time
                        dset_player_info_time[current_size_time] = player_id
                        dset_action_time[current_size_time] = (trump_time - time_last_action).total_seconds()
                        saved_states_time += 1

                        logging.debug(f"Player {player_id} sets trump to {trump_val}")
                        trump = trump_val
                        if trump == 0:
                            trump_pushed = True

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
                        card_played = int(m_play.group(3)) + 1

                        if player_id not in player_order:
                            logging.warning(f"Line {line_count}: Player {player_id} played card but player order is not known {player_order}.")
                            discarded = True
                            break

                        # creating shown cards in the order of player ids
                        card_shown = shown[:player_order[player_id]] + shown[player_order[player_id]+1:] # Exclude the current player
                        card_shown = card_shown[player_order[player_id]:] + card_shown[:player_order[player_id]] # Rotate to match player order

                        
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
                            logging.warning(f"Line {line_count}: Player {player_id} attempting to play card {card_played} but hand is recorded as empty. Skipping state save. {filename}")
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
                        current_points = player_points.get(player_id, 0)
                        previous_player = [uuid for uuid, value in player_order.items() if value == (player_order[player_id] + 1) % 4][0]
                        other_points = player_points.get(previous_player, 0)
                        current_history = list(history)
                        current_table_cards_only = list(current_trick_cards)

                        state = create_state(current_hand, current_table_cards_only, current_history, card_shown, trump)
                        # iterate over hands in player order to create the shown cards
                        shown_cards_cheated = [hands[pid] for pid in sorted(player_order, key=player_order.get)]
                        shown_cards_cheated = shown_cards_cheated[:player_order[player_id]] + shown_cards_cheated[player_order[player_id]+1:] # Exclude the current player
                        shown_cards_cheated = shown_cards_cheated[player_order[player_id]:] + shown_cards_cheated[:player_order[player_id]] # Rotate to match player order

                        state_cheated = create_state(current_hand, current_table_cards_only, current_history, shown_cards_cheated, trump)
                        state_with_points = create_state_with_points(current_hand, current_table_cards_only, current_history, card_shown, trump, current_points, other_points)

                        # --- Create Action ---
                        action = create_action(card_played, current_hand)

                        # --- Save State and Action to HDF5 ---

                        # Resize datasets before appending
                        current_size_play = dset_state_play.shape[0]
                        dset_state_play.resize(current_size_play + 1, axis=0)
                        dset_player_info_play.resize(current_size_play + 1, axis=0)
                        dset_action_play.resize(current_size_play + 1, axis=0)
                        dset_state_play_with_points.resize(current_size_play + 1, axis=0)
                        dset_action_play_with_points.resize(current_size_play + 1, axis=0)
                        dset_state_play_cheated.resize(current_size_play + 1, axis=0)
                        dset_action_play_cheated.resize(current_size_play + 1, axis=0)

                        current_size_time = dset_state_time.shape[0]
                        dset_state_time.resize(current_size_time + 1, axis=0)
                        dset_player_info_time.resize(current_size_time + 1, axis=0)
                        dset_action_time.resize(current_size_time + 1, axis=0)
                        dset_state_time[current_size_time:] = state

                        # Append data to the last row
                        dset_state_play_with_points[current_size_play:] = state_with_points
                        dset_action_play_with_points[current_size_play:] = [action]
                        dset_state_play_cheated[current_size_play:] = state_cheated
                        dset_action_play_cheated[current_size_play:] = [action]
                        dset_state_play[current_size_play:] = state
                        dset_player_info_play[current_size_play] = player_id
                        dset_action_play[current_size_play:] = action
                        saved_states_play += 1

                        dset_state_time[current_size_time:] = state
                        dset_player_info_time[current_size_time] = player_id
                        dset_action_time[current_size_time] = time_delta_seconds
                        saved_states_time += 1

                        # --- Update Game State AFTER saving ---
                        card_found_in_hand = False
                        for idx, hand_card in enumerate(hands[player_id]):
                            if hand_card == card_played:
                                del hands[player_id][idx]
                                card_found_in_hand = True
                                break
                        if not card_found_in_hand:
                            logging.warning(f"Line {line_count}: Card ({card_played}) played by {player_id} was not found in their current hand!")
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
                        if round_card_counter == 36:
                             logging.debug(f"--- Round {round_number} Complete ---")
                             # State resets automatically on next 'newRound' match

                    except (ValueError, IndexError, KeyError) as e:
                         logging.error(f"Line {line_count}: Failed to process card play for player {m_play.group(2)} card {m_play.group(3)} ({e}). Skipping.")
                         logging.debug(f"  Problematic line: {line}")
                         logging.exception("Traceback:")
                         discarded = True
                         break # Critical error, break out of loop
                    continue # Ensure we move to the next line even if errors 

                # Match Player Points (Game End)
                m_point = r_point.match(line)
                if m_point:
                    try:
                        point_time = datetime.strptime(m_point.group(1), "%Y-%m-%d %H:%M:%S%z")
                        player_id = m_point.group(2)
                        points = int(m_point.group(3))

                        if player_id not in player_ids_in_game:
                            logging.warning(f"Line {line_count}: Player {player_id} points but not in known players {player_ids_in_game}. Points ignored.")
                            continue

                        player_points[player_id] = points
                        logging.debug(f"Player {player_id} scored {points} points.")

                    except (ValueError, IndexError) as e:
                        logging.error(f"Line {line_count}: Error parsing player points ({e}). Skipping.")
                    continue
                
                # Match Game End
                m_end = r_gameEnd.match(line)
                if m_end:
                    logging.debug(f"Game ended")
                    break # End of file reached, break out of loop

    except FileNotFoundError:
        logging.error(f"File '{filename}' not found. Skipping.")
        discarded = True
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing file '{filename}' at line ~{line_count}: {e}")
        logging.exception("Traceback:")
        # Decide if you want to stop processing this file or continue
        discarded = True

    # Add final attributes after processing the file
    group_play.attrs['total_states_saved'] = saved_states_play
    group_play_with_points.attrs['total_states_saved'] = saved_states_play
    group_play_cheated.attrs['total_states_saved'] = saved_states_play
    group_trump.attrs['total_states_saved'] = saved_states_trump
    group_trump_with_points.attrs['total_states_saved'] = saved_states_trump
    group_time.attrs['total_states_saved'] = saved_states_time


    if not found_all_players and saved_states_play > 0:
         logging.warning(f"Finished processing {filename}, but player info might be incomplete. Saved {saved_states_play},{saved_states_time},{saved_states_trump} states.")
    elif saved_states_play > 0:
         logging.debug(f"Finished processing {filename}. Saved {saved_states_play},{saved_states_time},{saved_states_trump} state-action pairs to group /{game_group_name}.")
    else:
         logging.warning(f"Finished processing {filename}. No state-action pairs saved (possibly due to missing info or errors).")

    if discarded:
        logging.warning(f"Discarded file {filename} due to critical errors. No data saved.")
        del hdf5_play[game_group_name] # Remove the group from HDF5 file
        del hdf5_trump[game_group_name]
        del hdf5_time[game_group_name]
        del hdf5_play_with_points[game_group_name]
        del hdf5_trump_with_points[game_group_name]
        del hdf5_play_cheated[game_group_name]
        return (0, 0, 0) # Return zero states saved
    
    hdf5_play.flush()
    hdf5_trump.flush()
    hdf5_time.flush()
    hdf5_play_with_points.flush()
    hdf5_trump_with_points.flush()
    hdf5_play_cheated.flush()
    return (saved_states_play, saved_states_trump, saved_states_time)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Jass game log files and save state-action pairs to HDF5.")
    parser.add_argument("-i","--input-dir", required=True, help="Directory containing the .game log files.")
    parser.add_argument("-o","--output-dir", required=True, help="Path to the output directory.")
    parser.add_argument("-n","--num-files", type=int, default=0, help="Number of files to process (0 for all).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose mode enabled. Detailed logs will be shown.")

    play_file = os.path.join(args.output_dir, "playing.hdf5")
    trump_file = os.path.join(args.output_dir, "trump.hdf5")
    time_file = os.path.join(args.output_dir, "time.hdf5")
    play_file_with_points = os.path.join(args.output_dir, "playing_with_points.hdf5")
    play_file_cheated = os.path.join(args.output_dir, "playing_cheated.hdf5")
    trump_file_with_points = os.path.join(args.output_dir, "trump_with_points.hdf5")

    try:
        with h5py.File(play_file, 'a') as hdf5_playing, h5py.File(trump_file, 'a') as hdf5_trump, h5py.File(time_file, 'a') as hdf5_time, h5py.File(play_file_with_points, 'a') as hdf5_playing_with_points, h5py.File(trump_file_with_points, 'a') as hdf5_trump_with_points, h5py.File(play_file_cheated, 'a') as hdf5_playing_cheated:
            input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.game')]
            num_states_played = 0
            num_states_trump = 0
            num_states_time = 0
            if args.num_files > 0:
                input_files = input_files[:args.num_files]
            counter = 0
            for filename in tqdm.tqdm(input_files, desc="Processing files", unit="file"):
                file_path = os.path.join(args.input_dir, filename)
                states_played, states_trump, states_time = parse_file(file_path, hdf5_playing, hdf5_trump, hdf5_time, hdf5_playing_with_points, hdf5_trump_with_points, hdf5_playing_cheated)
                num_states_played += states_played
                num_states_trump += states_trump
                num_states_time += states_time
                counter += 1

            logging.info(f"Total states saved: {num_states_played}")

            hdf5_playing.attrs['total_states_saved'] = num_states_played
            hdf5_playing_with_points.attrs['total_states_saved'] = num_states_played
            hdf5_trump.attrs['total_states_saved'] = num_states_trump
            hdf5_trump_with_points.attrs['total_states_saved'] = num_states_trump
            hdf5_playing_cheated.attrs['total_states_saved'] = num_states_played
            hdf5_time.attrs['total_states_saved'] = num_states_time
    
    except Exception as e:
        # Catch errors related to opening/writing the HDF5 file itself
        logging.critical(f"Failed to open or write to HDF5 file {args.output_dir}: {e}")
        logging.exception("Traceback:")
