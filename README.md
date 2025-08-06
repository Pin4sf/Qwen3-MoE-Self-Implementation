# A From-Scratch Implementation of the Qwen3-MoE Model

This repository contains a from-scratch implementation of the **Qwen3 Mixture-of-Experts (MoE)** Large Language Model using PyTorch. The project offers a detailed, code-level exploration of a state-of-the-art sparse model architecture, which has been systematically refactored from a research-oriented notebook into a modular, production-ready Python package.

## 🚀 Project Objectives

The primary objective of this project is to provide a comprehensive, first-principles implementation of a contemporary Large Language Model. By constructing a sophisticated, sparse architecture such as Qwen3-MoE, this work aims to achieve the following:

* **Architectural Deconstruction**: To dissect and implement the core engineering and design principles that underpin modern high-performance generative models.

* **Conceptual Mastery**: To solidify a practical understanding of the Transformer architecture, its associated attention mechanisms, and advanced optimization strategies.

* **Theoretical to Practical Application**: To translate complex academic concepts, as detailed in the official Qwen3 technical report, into clean, functional, and efficient PyTorch code.

This project serves as a demonstration of the capacity to engineer and analyze complex AI systems, reflecting the deep learning and software engineering competencies required for advanced AI and machine learning roles.

## ✨ Core Features

The implementation faithfully recreates the key architectural components that define the Qwen3-MoE model's efficiency and performance.

* **🧠 Mixture-of-Experts (MoE) Layers**:
    The implementation includes sparse MoE layers that feature a **gating network**, or router, which dynamically selects a subset of "expert" feed-forward networks for each input token. This architectural paradigm permits a substantial increase in the model's parameter count while maintaining a constant computational cost (FLOPs) during inference.

* **⚡ Grouped-Query Attention (GQA)**:
    The GQA mechanism, a computationally efficient variant of Multi-Head Attention, has been constructed. In GQA, multiple query heads share a single key and value head, which significantly reduces the memory footprint of the Key-Value (KV) cache, thereby lowering memory bandwidth requirements and accelerating inference.

* **🌀 Rotary Position Embeddings (RoPE)**:
    RoPE has been coded from scratch to integrate relative positional information directly into the self-attention mechanism. This method is demonstrably more effective than traditional absolute position embeddings for processing long input sequences.

* **💾 KV Caching for Accelerated Inference**:
    A Key-Value cache has been developed to store the computed keys and values from preceding tokens. This crucial optimization prevents redundant computations in the attention layers, boosting text generation throughput by an order of magnitude.

* **🔧 Modular and Reusable Architecture**:
    The entire codebase has been refactored from a monolithic notebook into a modular structure (e.g., `layers`, `model`, `utils`), adhering to software engineering best practices for building maintainable and scalable systems.

* **🤗 Hugging Face Integration**:
    The model is designed to seamlessly load official, pre-trained Qwen3 weights directly from the Hugging Face Hub, ensuring compatibility and enabling practical application to real-world text generation tasks.

## 📂 Project Structure

The repository is organized using a clean, modular structure to enforce separation of concerns and enhance code readability.

```

qwen3-from-scratch/
├── notebooks/              \# Original Jupyter Notebook for exploration
├── scripts/                \# User-facing command-line interface scripts
│   ├── download\_model.py   \# Script for downloading model weights
│   └── generate\_text.py    \# Script for text generation execution
├── src/                    \# Core source code of the model
│   ├── layers.py           \# Fundamental model layers (MoE, RMSNorm, etc.)
│   ├── model.py            \# Main Qwen3Model architectural definition
│   ├── tokenizer.py        \# Custom Qwen3 tokenizer class
│   ├── kv\_cache.py         \# Key-Value Cache implementation
│   └── utils.py            \# Helper functions (weight loading, config)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```

## ⚙️ Setup and Installation

1.  **Cloning the Repository:**
    ```bash
    git clone [https://github.com/your-username/qwen3-from-scratch.git](https://github.com/your-username/qwen3-from-scratch.git)
    cd qwen3-from-scratch
    ```

2.  **Creating a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Installing Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💡 Usage Guide

### 1. Downloading Model Weights

Utilize the provided script to download the model weights and tokenizer from the Hugging Face Hub. By default, the files will be stored in the `models/` directory.

```bash
# Example: Download the Qwen3 Coder 30B MoE model
python scripts/download_model.py --repo_id "Qwen/Qwen3-Coder-30B-A3B-Instruct"
```

### 2. Generating Text

Execute the generation script with a specified prompt. The script will automatically load the corresponding model and tokenizer, then stream the generated text to the console.

```bash
python scripts/generate_text.py --prompt "Write a Python function to find the nth Fibonacci number using recursion."
```

#### Command-Line Options:

  * `--prompt`: (Required) The input text prompt to initiate generation.
  * `--repo_id`: The Hugging Face repository ID of the model to be utilized.
  * `--max-new-tokens`: The maximum number of new tokens to be generated.
  * `--no-kv-cache`: A flag to disable the KV Cache, primarily for performance comparison.

## ⚡ Performance Analysis: The Efficacy of KV Caching

The **Key-Value (KV) Cache** represents a critical optimization for autoregressive transformer models. By storing the intermediate attention values (Keys and Values) for all tokens in the context, the need to re-compute these values at each generation step is obviated. This optimization effectively reduces the computational complexity of processing each new token from being dependent on the sequence length to being constant.

The result is a substantial improvement in inference throughput, which makes real-time text generation computationally feasible. The performance impact of this mechanism can be observed by executing the generation script with and without the `--no-kv-cache` flag.

## 🙏 Acknowledgments

This implementation acknowledges the influence of the excellent educational materials provided by **Sebastian Raschka** in his book, [*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv), and the technical specifications detailed in the official [**Qwen3 Technical Report**](https://arxiv.org/abs/2505.09388). The project's structure and code are adapted from the standalone notebook that accompanies the book's supplementary materials.