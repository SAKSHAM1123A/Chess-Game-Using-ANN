# ANN Chess Engine

A chess engine that uses a neural network (ANN) to evaluate positions, combined with minimax search and alpha-beta pruning to choose moves. Built in Google Colab with PyTorch and python-chess.

## How it works

1. **Board encoding:** each position becomes a vector of 769 numbers: 12 planes of 64 squares (6 piece types x 2 colors) plus 1 value for the side to move.
2. **Evaluator network:** a feed-forward neural network predicts a score from -1 (Black winning) to +1 (White winning).
3. **Training data:** positions sampled from real games, labelled with the final game result.
4. **Search:** minimax with alpha-beta pruning calls the network to pick the best move. `python-chess` handles the rules and legal moves.

## Dataset

[Chess Games by arevel (Kaggle)](https://www.kaggle.com/datasets/arevel/chess-games), downloaded with `kagglehub`.

## Files

```
chess_game.py   # full pipeline: data loading, training and playing
README.md
```

## Run on Google Colab (recommended)

1. Upload `chess_game.py` to Colab, or paste its contents into notebook cells.
2. Install the dependencies in a cell:
```
   !pip install chess kagglehub torch numpy pandas
```
3. Run the script. It downloads the dataset, trains the network and starts the game.

## Run locally

```bash
pip install chess kagglehub torch numpy pandas
python chess_game.py
```

The dataset is downloaded through `kagglehub`, so you need to be able to authenticate with Kaggle on your machine. If the script uses Colab-only paths such as `/content/...`, change them to local paths first.

## Results

[Add your own numbers here, such as validation loss and how the engine plays against a simple material-counting engine.]

## Limitations

- The network is trained on game outcomes, not engine evaluations, so its labels are noisy.
- The search is pure Python with no quiescence search, so it is slow and can misjudge tactics. Use a low search depth.
- This is an educational project, not a strong engine.

## Future improvements

- Train on Stockfish-evaluated positions.
- Add move ordering and a transposition table.
- Try a convolutional network on the 8x8 board.
- Add a graphical interface.

## Acknowledgements

- [python-chess](https://github.com/niklasf/python-chess)
- [Chess Games dataset by arevel](https://www.kaggle.com/datasets/arevel/chess-games)
- [PyTorch](https://pytorch.org/)

## License

MIT (add a `LICENSE` file when you create the repo)
