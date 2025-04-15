# State

## Input 

The input comprises all information currently available for the player, as well as the card history of the current round.
The decision has been taken to exclude the score, on the basis that it is considered irrelevant to gameplay.

### Input Breakdown

1. **History of current round:**  
   - There are 32 cards, each represented by a 13-bit vector.  
   - Total bits for history: 32 × 13 = **416 bits**

2. **Current Cards Played:**  
   - There are 3 cards, each with a 13-bit vector.  
   - Total bits for current cards played: 3 × 13 = **39 bits**

3. **Current Cards in Hand:**  
   - There are 9 cards, each represented by a 13-bit vector.  
   - Total bits for the hand: 9 × 13 = **117 bits**

4. **Trump Information:**  
   - Represented by a 6-bit vector (typically one-hot encoding for the 4 possible trump suits).  
   - Total bits for trump: **6 bits**
   - Diamonds ♦, Clubs ♣, Hearts ♥, Spades ♠, Top down ↓, Bottom up ↑

```
+--------------------------------------------------+
|              Input State (578 bits)              |
+--------------------------------------------------+
|  History (32 cards x 13 bits)         | 416 bits |  
|  [ C1 ][ C2 ][ C3 ] ... [ C32 ]       | (32x13)  |
+--------------------------------------------------+
|  Current Cards Played (3 x 13)        | 39 bits  |
|  [ C1 ][ C2 ][ C3 ]                   | (3x13)   |
+--------------------------------------------------+
|  Current Cards in Hand (9 x 13)       | 117 bits |
|  [ C1 ][ C2 ][ C3 ] ... [ C9 ]        | (9x13)   |
+--------------------------------------------------+
|  Current Cards in other (4 x 9 x 13)  | 468 bits |
|  [ C1 ][ C2 ][ C3 ] ... [ C9 ]        | (4x9x13) |
+--------------------------------------------------+
|  Trump Suit                           | 6 bits   |
|  [ TTTTTT ]                           | (6)      |
+--------------------------------------------------+
```

That would give us a input size of: 
416 (History) + 39 (Current Played) + 117 (Hand) + 468 (other hands) + 6 (Trump) = **1046 bits**

So, the complete input state is a 1046-dimensional vector.

### Encoding Details

- **12-Bit Vector Card:**  
  Each card is encoded with 12 bits:
  - The **first 4 bits** encode the suit. (from msb)
  - The **remaining 8 bits** encode the card's value.

   ```
   +-------------------------+
   |    Card (13 bits)       |
   +-------------------------+
   |  4 bits: Suit           |  --> One-hot encoded
   +-------------------------+
   |  9 bits: Value          |  --> One-hot encoded
   +-------------------------+
   ```
  
  This is effectively a one-hot encoding for both suit and value combined.

   ```
   +----------------------+
   | 12 bit: Suit Spades  |
   +----------------------+
   | 11 bit: Suit Clubs   |
   +----------------------+
   | 10 bit: Suit Diamonds|
   +----------------------+
   |  9 bit: Suit Hearts  |
   +----------------------+
   |  8 bit: Value Ace    |
   +----------------------+
   |  7 bit: Value King   |
   +----------------------+
   |  6 bit: Value Quen   |
   +----------------------+
   |  5 bit: Value Jack   |
   +----------------------+
   |  4 bit: Value 10     |
   +----------------------+
   |  3 bit: Value 9      |
   +----------------------+
   |  2 bit: Value 8      |
   +----------------------+
   |  1 bit: Value 7      |
   +----------------------+
   |  0 bit: Value 6      |
   +----------------------+
   ```
- **6-Bit Vectore Trump:**
   Trump is endcode with 6 bits:
   ```
   +----------------------+    
   | 5 bit: Trump Topdown |
   +----------------------+
   | 4 bit: Trump Bottomup|
   +----------------------+
   | 3 bit: Trump Spades  |
   +----------------------+
   | 2 bit: Trump Clubs   |
   +----------------------+
   | 1 bit: Trump Diamonds|
   +----------------------+
   | 0 bit: Trump Hearts  |
   +----------------------+
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