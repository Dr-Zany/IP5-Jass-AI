# State

## Input 

The input comprises all information currently available for the player, as well as the card history of the current round.
The decision has been taken to exclude the score, on the basis that it is considered irrelevant to gameplay.

### Input Breakdown

1. **History of current round:**  
   - There are 32 cards, each represented by a 36 dimensions vector.
   - The vector is one hot.
   - Total dimensions for history: 32 × 36 = **416**

2. **Current Cards Played:**  
   - There are 3 cards, each represented as 36 dimensions vector.
   - The vector is one hot.
   - Total dimensions for current cards played: 3 × 36 = **108**

3. **Current Cards in Hand:**  
   - There are 9 cards, each represented by a 36 dimensions vector.  
   - The vector is one hot.
   - Total dimensions for the hand: 9 × 36 = **324**

4. **Current Cards of Others**
   - There are 3 other players, where we can know up to 9 cards, each represented as 36 dimensions vector.
   - The vector is one hot.
   - Total dimensions for other players: 3 × 9 × 36 = **972**
   - **Note:** The information of the other players cards comes from so called "weissen".

4. **Trump Information:**  
   - Represented by a 6 dimensions vector.
   - The vector is one hot.
   - Total dimensions for trump: **6**
   - Diamonds ♦, Clubs ♣, Hearts ♥, Spades ♠, Top-down ↓, Bottom-up ↑

That would give us a input dimension of: 
1152 (History) + 108 (Current Played) + 324 (Hand) + 973 (other hands) + 6 (Trump) = **2527**
By using a embedding of 36 dimensions for each card, we can represent the cards as indexs form.

```
+---------------------------------------------------------+
|                Input State (72)                         |
+---------------------------------------------------------+
|  History Cards 32                       | Value 0 -> 36 |
|  [ C1 ][ C2 ][ … ][ C32 ]               |               |
+---------------------------------------------------------+
|  Current Cards Played 3                 | Value 0 -> 36 |
|  [ C1 ][ C2 ][ C3 ]                     |               |
+---------------------------------------------------------+
|  Current Cards in Hand 9                | Value 0 -> 36 |
|  [ C1 ][ C2 ][ … ][ C9 ]                |               |
+---------------------------------------------------------+
|  Current Cards of Others 27             | Value 0 -> 36 |
|  [ C1 ][ C2 ][ … ][ C9 ] × 3 players    |               |
+---------------------------------------------------------+
|  Trump Suit 1                           | Value 1 -> 6  |
|  [ ↓  ↑  ♠  ♥  ♣  ♦ ]                   |               |
+---------------------------------------------------------+

```
The input state can be represented as 72 indices, where each index corresponds to a specific card or trump suit.

So, the complete input state is a 2527-dimensional vector.

### Encoding Details

- **36 dimensions Card Vector:**  
  Each card is encoded in a 36-dimensional vector, where each dimension corresponds to a specific card. The encoding is as follows:
   ```
   +-----------------------------+
   |  36 - 28:   Spades   ♠      |
   |  [ A ][ K ][ Q ][ … ][ 6 ]  |
   +-----------------------------+
   |  27 - 19:   Hearts   ♥      |
   |  [ A ][ K ][ Q ][ … ][ 6 ]  |
   +-----------------------------+
   |  18 - 10:   Clubs    ♣      |
   |  [ A ][ K ][ Q ][ … ][ 6 ]  |
   +-----------------------------+
   |   9 -  1:   Diamonds ♦      |
   |  [ A ][ K ][ Q ][ … ][ 6 ]  |
   +-----------------------------+
   |        0:   None            |
   +-----------------------------+
   ```

   - Each card is represented by a one-hot encoding, where only one dimension is set to 1 (indicating the presence of that card), and all other dimensions are set to 0.
   - The order of cards from Ace to 6 means Ace is the highest dimension and 6 is the lowest dimension. 

- **6-Bit Vectore Trump:**
   Trump is endcode with 6 bits:
   ```
   +------------------+    
   | 6:  Top-down  ↓  |
   +------------------+
   | 5:  Bottom-up ↑  |
   +------------------+
   | 4:  Spades    ♠  |
   +------------------+
   | 3:  Hearts    ♥  |
   +------------------+
   | 2:  Clubs     ♣  |
   +------------------+
   | 1:  Diamonds  ♦  |
   +------------------+
   ``` 

### Output

**Card to Play**
- The network outputs **9 softmaxed floating-point values**.
- Each of these corresponds to a card in hand.
- Before applying softmax, the output is **masked** with legal moves (i.e., illegal moves are zeroed out).

**Time prediction**
- The network outputs value in Secoends howlong it should wait before playing

### Summary

- **Total Input Dimension:** 532 (from 384 + 36 + 108 + 4)
- **Output:** 9 probabilities (one per card in hand), with a softmax applied over legal moves.

# Action

## Output
The output of the model is a 9-dimensional vector, where each dimension corresponds to a card in hand. The values in this vector represent the probabilities of playing each card.

# Dataset

## Structure

The hdf5 structured in to multiples group. each group has the same name as the log file it comes from.
in each group there are multiple datasets: state, action, time, player_id:
- **state:** The state of the game at the time of the action.
- **action:** The action taken by the player at that time.
- **time:** The time taken by the player to make the action.
- **player_id:** The id of the player who took the action.

```
jass_datset.hdf5/
├─ S2500_0000be8243f6ebc8e254c91b4a3f5b59/
│  ├─ state
│  ├─ action
│  ├─ player_id
│  ├─ time
├─ S2500_000c61fbfdde242d0a30e61545fa863a/
¦  ├─ state
¦  ├─ action
¦  ├─ player_id
¦  ├─ time
├─ S2500_00e900febb7ba2fb67c8730df9c8d25f/
   ├─ state
   ├─ action
   ├─ player_id
   ├─ time
```

### State
- The state is a 72 indices structure like the input state.