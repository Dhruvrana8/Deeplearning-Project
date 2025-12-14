# Character-Level Language Model for Name Generation

This project implements a character-level language model using a custom neural network architecture to generate names. The model learns to predict the next character in a sequence given a context of previous characters.

## Overview

- **Goal**: Generate new names character by character.
- **Architecture**: Neural network with embeddings, linear layers, and batch normalization.
- **Dataset**: A list of names (synthetic or real).

## Files Description

- **`index.ipynb`**: The main notebook that contains the model architecture, training loop, and evaluation code. It uses PyTorch for the model and MLflow/WandB for experiment tracking.
- **`make_synthetic_data.ipynb`**: A utility notebook used to preprocess raw name data or generate synthetic datasets for training.
- **`synthetic_data.txt`**: (Generated) The processed dataset used for training.

## Getting Started

1.  **Install Dependencies**:
    Ensure you have Python installed along with the required libraries.

    ```bash
    pip install torch matplotlib mlflow numpy
    ```

2.  **Prepare Data**:
    Run `make_synthetic_data.ipynb` to clean or generate the `synthetic_data.txt` file from your raw data source (e.g., `name.txt`).

3.  **Train the Model**:
    Open `index.ipynb` and run the cells to:
    - Load the dataset.
    - Initialize the model.
    - Train the model using the training loop.
    - Generate new names using the trained model.
