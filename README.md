# Deep Learning Project: Character-Level Language Models

A comprehensive deep learning project exploring character-level language models for name generation, from simple bigram models to advanced neural network architectures with embeddings and batch normalization.

## 🎯 Project Overview

This repository contains multiple implementations of character-level language models, progressively increasing in complexity:

1. **Statistical Bigram Model** - Count-based probability distributions
2. **Single-Layer Neural Network** - Basic gradient descent learning
3. **Multi-Layer Perceptron (MLP)** - Advanced architecture with embeddings and batch normalization

The project demonstrates fundamental deep learning concepts including gradient descent, backpropagation, embeddings, batch normalization, and various optimization techniques.

## 📁 Repository Structure

```
deeplearning-project/
├── makemore/                    # Bigram and basic neural network models
│   ├── Build_make_more.ipynb   # Statistical and single-layer NN models
│   ├── build_makemore_mlp.ipynb # Multi-layer perceptron implementation
│   └── build_makemore_mlp_part_5.ipynb
├── project/                     # Advanced character-level model
│   ├── index.ipynb             # Main training notebook with experiment tracking
│   ├── make_synthetic_data.ipynb # Data preprocessing utilities
│   └── README.md               # Detailed project documentation
├── data/                        # Training datasets
│   ├── name.txt                # Original names dataset (32,033 names)
│   └── synthetic_data.txt      # Processed training data
├── assets/                      # Project screenshots and visualizations
├── Build_make_more.md          # Detailed documentation of bigram models
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

## 🚀 Features

### Statistical Bigram Model

- Character-level probability distributions
- Model smoothing to prevent zero probabilities
- Negative Log-Likelihood (NLL) evaluation (~2.454 loss)
- Visualization of bigram count matrices

### Neural Network Models

- **Single-Layer Network**: Learns bigram probabilities via gradient descent
- **Multi-Layer Perceptron**:
  - Character embeddings
  - Multiple hidden layers
  - Batch normalization
  - Advanced optimization (Adam, RMSprop, SGD)
  - Experiment tracking with MLflow and Weights & Biases

### Name Generation

- Sample-based generation from learned distributions
- Configurable context window
- Temperature-controlled sampling

## 📊 Dataset

- **Source**: `name.txt` containing 32,033 unique names
- **Vocabulary**: 27 characters (a-z + special token)
- **Preprocessing**: Bigram extraction with start/end tokens
- **Statistics**:
  - Shortest name: 2 characters
  - Longest name: 15 characters

## 🛠️ Installation

### Prerequisites

- Python >= 3.11
- pip or uv package manager

### Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd Deeplearning-Project
   ```

2. **Install dependencies**:

   Using pip:

   ```bash
   pip install torch matplotlib mlflow wandb numpy python-dotenv
   ```

   Or using uv (recommended):

   ```bash
   uv sync
   ```

3. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your MLflow/WandB credentials
   ```

## 📖 Usage

### 1. Statistical Bigram Model

Navigate to the `makemore` directory and open `Build_make_more.ipynb`:

```bash
jupyter notebook makemore/Build_make_more.ipynb
```

This notebook demonstrates:

- Building count-based probability matrices
- Calculating NLL loss
- Generating names from statistical distributions

### 2. Neural Network Bigram Model

In the same notebook, explore the neural network section:

- One-hot encoding of characters
- Weight matrix learning via gradient descent
- Comparison with statistical model

### 3. Advanced MLP Model

Open `makemore/build_makemore_mlp.ipynb` for:

- Character embeddings
- Multi-layer architecture
- Batch normalization experiments
- Optimizer comparisons (SGD, Adam, RMSprop)

### 4. Production Model with Experiment Tracking

Navigate to the `project` directory:

```bash
cd project
jupyter notebook index.ipynb
```

Features:

- MLflow experiment tracking
- Weights & Biases integration
- Hyperparameter logging
- Model checkpointing

### 5. Data Preprocessing

To create custom datasets:

```bash
jupyter notebook project/make_synthetic_data.ipynb
```

## 🧪 Experiments & Results

### Model Performance

| Model Type         | Architecture               | Loss (NLL) | Notes                         |
| ------------------ | -------------------------- | ---------- | ----------------------------- |
| Statistical Bigram | Count-based                | ~2.454     | Baseline performance          |
| Single-Layer NN    | 27x27 weights              | ~2.457     | Matches statistical model     |
| MLP                | Embeddings + Hidden Layers | Varies     | Best with batch normalization |

### Key Findings

- **Batch Normalization**: Significantly improves convergence speed and stability
- **Optimizer Comparison**: Adam typically outperforms SGD and RMSprop for this task
- **Embedding Dimension**: Larger embeddings capture more character relationships
- **Context Window**: Longer context improves generation quality

## 📚 Learning Objectives

This project covers:

1. **Probability & Statistics**

   - Bigram probability distributions
   - Smoothing techniques
   - Log-likelihood evaluation

2. **Neural Networks**

   - Gradient descent optimization
   - Backpropagation
   - Weight initialization
   - Activation functions

3. **Deep Learning Techniques**

   - Character embeddings
   - Batch normalization
   - Optimizer comparison
   - Regularization

4. **MLOps & Experiment Tracking**
   - MLflow integration
   - Weights & Biases logging
   - Hyperparameter management
   - Reproducibility

## 🔬 Advanced Topics

### Batch Normalization

The project includes detailed experiments on batch normalization:

- Training vs. inference behavior
- Impact on convergence speed
- Accuracy improvements
- Proper usage of `model.eval()` and `model.train()`

See [Build_make_more.md](Build_make_more.md) for detailed documentation.

### Optimizer Analysis

Comprehensive comparison of optimization algorithms:

- **SGD**: Simple but requires careful learning rate tuning
- **Adam**: Adaptive learning rates, generally best performance
- **RMSprop**: Good for non-stationary objectives

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional model architectures (LSTM, Transformer)
- More sophisticated generation techniques
- Extended datasets
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Andrej Karpathy's "makemore" series
- Dataset sourced from publicly available name lists
- Built with PyTorch, MLflow, and Weights & Biases

## 📞 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This is an educational project demonstrating fundamental deep learning concepts. For production name generation, consider more advanced architectures like GPT or LSTM-based models.
