# Movie Review Sentiment Analysis

A simple **RNN-based sentiment analysis** project that classifies IMDB movie reviews as **Positive** or **Negative** using a trained TensorFlow/Keras model and a Streamlit web app.

## Project Overview

This project:

- trains a **SimpleRNN** model on the IMDB movie review dataset
- uses **embedding layers** to convert words into dense vectors
- classifies reviews as:
  - **Positive**
  - **Negative**
- provides a **Streamlit UI** for entering custom reviews

## Files

- `simplelearn.ipynb` — notebook used to load data, preprocess text, build, train, and save the model
- `main.py` — Streamlit app for sentiment prediction
- `simple_rnn_imdb.h5` — saved trained model
- `pyproject.toml` — project dependencies

## Model Architecture

The model uses:

- `Embedding(max_features, 128)`
- `SimpleRNN(128, activation='relu')`
- `Dense(1, activation='sigmoid')`

## Dataset

The project uses the built-in **IMDB dataset** from `tensorflow.keras.datasets.imdb`.

- Vocabulary size: `10000`
- Input sequence length: `500`

## Requirements

- Python `3.13+`
- TensorFlow
- NumPy
- Streamlit
- Jupyter

## Installation

Install dependencies from the project folder:

```bash
uv sync
```

If `uv` is not available, use:

```bash
pip install -r requirements.txt
```

## Training the Model

Open `simplelearn.ipynb` and run the cells in order:

1. Load the IMDB dataset
2. Pad sequences to length `500`
3. Build the SimpleRNN model
4. Compile the model
5. Train with early stopping
6. Save the model as `simple_rnn_imdb.h5`

## Running the App

Start the Streamlit app from the project directory:

```bash
streamlit run main.py
```

Then open the local URL shown in the terminal.

## Usage

1. Paste a movie review into the text box
2. Click **Classify**
3. View the predicted sentiment and score

## Notes

- The IMDB dataset is tokenized into integers before training.
- The app uses the saved model file `simple_rnn_imdb.h5`.
- The review text is lowercased and converted to token IDs before prediction.

## Example

Input:

> This movie was amazing and very emotional.

Output:

> Sentiment: Positive

## License

For learning and educational use.