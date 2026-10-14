# Chess-Game-Using-ANN
Chess engine that uses a neural network (ANN) to evaluate positions, combined with minimax and alpha-beta search. Built with PyTorch and python-chess.

# ANN Chess Engine

A chess engine where an artificial neural network (ANN) decides how good a position is, and a minimax search with alpha-beta pruning uses that score to choose moves. You can play against it in the terminal.

## How it works

1. **Board encoding:** each position becomes a vector of 769 numbers: 12 planes of 64 squares (6 piece types x 2 colors) plus 1 value for the side to move.
2. **Evaluator network:** a feed-forward network (769 → 512 → 128 → 32 → 1, ReLU, tanh output) predicts a score between -1 (Black winning) and +1 (White winning).
3. **Training data:** positions sampled from real games, each labelled with the final game result.
4. **Search:** negamax with alpha-beta pruning calls the network at the leaves to pick the best move. Move legality is handled by `python-chess`, so the network never has to produce a legal move itself.

## Dataset

[Chess Games by arevel (Kaggle)](https://www.kaggle.com/datasets/arevel/chess-games), Lichess games with moves and results.
Only games where both players are rated 2000+ are used, and positions from the first moves of each game are skipped.

## Project structure

```
encode.py     # board -> 769-number vector
prepare.py    # builds data.npz from the Kaggle dataset
train.py      # defines and trains the network, saves model.pt
engine.py     # alpha-beta search using the network
play.py       # play against the engine in the terminal
```

## Setup

```bash
git clone https://github.com/<your-username>/ann-chess-engine.git
cd ann-chess-engine
pip install chess numpy pandas torch kagglehub
```

## Usage

```bash
# 1. Download the dataset and set the path inside prepare.py
python -c "import kagglehub; print(kagglehub.dataset_download('arevel/chess-games'))"

# 2. Build the training data
python prepare.py

# 3. Train the network (saves model.pt)
python train.py

# 4. Play (you are White, enter moves like e4 or Nf3)
python play.py
```

## Results

[Add your own numbers here, for example validation loss, and how the engine does against a simple material-counting engine over N games.]

## Limitations

- The network is trained on game outcomes, not engine evaluations, so its labels are noisy.
- The search is written in Python and has no quiescence search, so it is slow and can misjudge tactics. Use depth 2-3.
- This is an educational project, not a strong engine.

## Ideas for improvement

- Train on Stockfish-evaluated positions (Lichess evaluation database).
- Add move ordering, quiescence search and a transposition table.
- Try a convolutional network on the 8x8 board.
- Add a Pygame graphical interface.

## Acknowledgements

- [python-chess](https://github.com/niklasf/python-chess)
- [Chess Games dataset by arevel](https://www.kaggle.com/datasets/arevel/chess-games)
- [PyTorch](https://pytorch.org/)

## License

MIT (add a `LICENSE` file when you create the repo)
