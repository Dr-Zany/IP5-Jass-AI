import argparse
from datetime import datetime
import h5py
import re
import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple

NUM_CARDS_HAND = 9
NUM_CARDS_TABLE = 3
NUM_CARDS_HISTORY = 32
NUM_CARDS_SHOWN = 27 # 9 cards shown for each player 
NUM_TRUMPS = 1

NUM_STATE = NUM_CARDS_HAND + NUM_CARDS_TABLE + NUM_CARDS_HISTORY + NUM_CARDS_SHOWN + NUM_TRUMPS # 1 for trump, 9 for each player, 32 for history, 27 for shown cards

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(thread)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler("jass_log_parser.log"), # Log to a file
                        logging.StreamHandler() # Also log to console
                    ])

# --- Regex Definitions ---
r_playerInfo = re.compile(r'\"usernickname\":\s*\"([\w\d]+)\".*?\"eq\":\s*(\d+).*?\"iq\":\s*(\d+).*?\"niceness\":\s*(\d+).*?\"honness\":\s*(\d+).*?\"winness\":\s*(\d+).*?\"playedgames\":\s*(\d+).*?\"profival\":\s*(\d+)', re.IGNORECASE)
r_newRound = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"action\":{\"newRound\": \"Runde (\d+)\"', re.IGNORECASE)
r_cardsDealt = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"action\":{\"doDeal\":\s*(\d+),\"player\":\s*\d+,\s*\"usernickname\":\s*\"([\w\d]+)\",\"cardset\":\s*\[\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\",\"(\d+)\"\]', re.IGNORECASE)
r_setTrump = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":{\"submitTrump\":\s*(-?\d+)', re.IGNORECASE)
r_playCard = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":{\"submitsCard\":\s*\"(\d+)\"', re.IGNORECASE)
r_cardShow = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":{\"doRequestSuits\":\s*\[".*?;((?:\d+,?)+)"',re.IGNORECASE)
r_gameEnd = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{4})\s.*\"pid\":\s*\"([\w\d]+)\",\"action\":\s*\"gameFinished\"}', re.IGNORECASE)

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
    for shown_cards in shown:
        state += shown_cards + [0] * (9 - len(shown_cards))
    state += [trump] # Add trump value
    if len(state) != NUM_STATE:
        logging.error(f"State length mismatch: expected {NUM_STATE}, got {len(state)}")
        raise ValueError(f"State length mismatch: expected {NUM_STATE}, got {len(state)}")

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
def parse_file(file_path: str, hdf5_file: h5py.File) -> int:
    """
    Parse the Jass log file and store the data in an HDF5 file.

    Args:
        file_path (str): Path to the Jass log file.
        hdf5_file (h5py.file): HDF5 file object to store the parsed data.
    """

    file_name = os.path.basename(file_path)
    game_group_name = os.path.splitext(file_name)[0] 

    player_infos: List[Dict[str, Any]] = []
    player_ids_in_game: List[str] = []
    found_all_players = False # Flag to track if player info is complete for this game

    if game_group_name in hdf5_file:
        logging.warning(f"Group '{game_group_name}' already exists in HDF5 file. Overwriting.")
        del hdf5_file[game_group_name]
    group = hdf5_file.create_group(game_group_name)
    logging.debug(f"Created HDF5 group: /{game_group_name}")


    dset_state = group.create_dataset("state", (0, NUM_STATE), maxshape=(None, NUM_STATE), chunks=True, dtype='uint8', compression="gzip")
    dset_player_info = group.create_dataset("player_id", (0,1), maxshape=(None, 1), chunks=True, dtype='uint8', compression="gzip")

    dset_action = group.create_dataset("action", (0,1), maxshape=(None, 1), chunks=True, dtype='uint8', compression="gzip")
    dest_time = group.create_dataset("time", (0,1), maxshape=(None, 1), chunks=True, dtype='float', compression="gzip")

    time_last_action: Optional[datetime] = None
    player_order: Dict[str, int] = {} # Maps player IDs to their order in the game
    hands: Dict[str, List[int]] = {} # Current cards in each player's hand
    shown: List[List[int]] = {} # Cards shown by each player (so called "Weiss")
    history: List[int] = [] # Cards from completed tricks in the current round
    table: List[Tuple[str, int]] = [] # Cards on table in the current trick [(player_id, card), ...]
    current_trick_cards: List[int] = [] # Just the cards on table for state creation
    round_number: int = 0
    trick_counter: int = 0 # Counts cards played in the current trick (0-3)
    round_card_counter: int = 0 # Counts total cards played in the round (0-35)
    trump: int = 0 # Current trump suit

    logging.debug(f"Processing file: {filename}")
    line_count = 0
    saved_states = 0

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
                    trump = 0
                    player_order = {}
                    shown = [[] for _ in range(4)]
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

                        if player_id not in player_ids_in_game:
                             logging.warning(f"Line {line_count}: Player {player_id} set trump but not in known players {player_ids_in_game}. Trump setting ignored.")
                             continue

                        logging.debug(f"Player {player_id} sets trump to {trump_val}")
                        trump = trump_val

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
                        current_history = list(history)
                        current_table_cards_only = list(current_trick_cards)

                        state = create_state(current_hand, current_table_cards_only, current_history, card_shown, trump)

                        # --- Create Action ---
                        action = create_action(card_played, current_hand)

                        # --- Save State and Action to HDF5 ---

                        # Resize datasets before appending
                        current_size = dset_state.shape[0]
                        dset_state.resize(current_size + 1, axis=0)
                        dset_player_info.resize(current_size + 1, axis=0)
                        dset_action.resize(current_size + 1, axis=0)
                        dest_time.resize(current_size + 1, axis=0)

                        # Append data to the last row
                        dset_state[current_size:] = state
                        dset_player_info[current_size] = list(hands.keys()).index(player_id)
                        dset_action[current_size:] = action
                        dest_time[current_size] = time_delta_seconds

                        saved_states += 1

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
        del hdf5_file[game_group_name] # Remove the group from HDF5 file
        return 0 # Skip this file
    
    hdf5_file.flush() # Ensure changes are written to disk
    return saved_states # Return the number of states saved for this file


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Jass game log files and save state-action pairs to HDF5.")
    parser.add_argument("-i","--input-dir", required=True, help="Directory containing the .game log files.")
    parser.add_argument("-o","--output-file", required=True, help="Path to the output HDF5 file.")
    parser.add_argument("-n","--num-files", type=int, default=0, help="Number of files to process (0 for all).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose mode enabled. Detailed logs will be shown.")

    try:
        with h5py.File(args.output_file, 'a') as hdf5_file:
            input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.game')]
            num_states = 0
            if args.num_files > 0:
                input_files = input_files[:args.num_files]
            counter = 0
            for filename in input_files:
                file_path = os.path.join(args.input_dir, filename)
                num_states += parse_file(file_path, hdf5_file)
                counter += 1
                if counter % 100 == 0:
                    logging.info(f"Processed {counter} files of {len(input_files)}. Total states saved so far: {num_states}")

            logging.info(f"Total states saved: {num_states}")

            hdf5_file.attrs['total_states_saved'] = num_states
    
    except Exception as e:
        # Catch errors related to opening/writing the HDF5 file itself
        logging.critical(f"Failed to open or write to HDF5 file {args.output_file}: {e}")
        logging.exception("Traceback:")
