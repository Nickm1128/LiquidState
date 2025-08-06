# Sparse Sine-Activated Liquid State Machine for Next-Token Prediction

This project implements a novel neural architecture that combines **Liquid State Machines (LSM)** with **Convolutional Neural Networks (CNN)** for next-token prediction on dialogue data. The system uses sparse connectivity patterns and parametric sine activation functions to create complex temporal dynamics, which are then encoded as 2D "waveforms" for CNN processing.

## Overview

### What is a Liquid State Machine?
A Liquid State Machine is a type of recurrent neural network inspired by biological neural circuits. Unlike traditional RNNs, LSMs maintain a "reservoir" of randomly connected neurons that create rich temporal dynamics. The key innovation in this project is:

1. **Sparse Connectivity**: Only a fraction of connections exist between neurons, making the network more efficient and biologically plausible
2. **Parametric Sine Activation**: Instead of standard activations, we use learnable sine functions: `A * exp(-α * |x|) * sin(ω * x)`
3. **Rolling Wave Encoding**: LSM outputs are encoded as 2D spatial-temporal patterns
4. **CNN Processing**: A CNN learns to interpret these 2D patterns for next-token prediction

### Architecture Components

1. **Data Loader**: Downloads and processes HuggingFace Synthetic-Persona-Chat dataset
2. **Sparse Reservoir**: Custom Keras layers with learnable sparse connectivity
3. **Rolling Wave Buffer**: Converts temporal dynamics to 2D waveforms
4. **CNN Model**: Processes waveforms to predict next token embeddings
5. **Training Pipeline**: Integrates all components for end-to-end learning

## Installation

### Requirements
- Python 3.9+
- TensorFlow 2.10+
- CUDA support recommended for GPU acceleration

### Setup
1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
