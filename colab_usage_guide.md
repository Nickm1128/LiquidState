# Google Colab Usage Guide for Advanced LSM System

## Getting Started

### 1. Clone the Repository
```bash
!git clone <your-repo-url>
%cd your-repo-name
```

### 2. Install Dependencies
```bash
!pip install tensorflow>=2.10 numpy pandas scikit-learn requests
```

### 3. Basic Training Commands

#### Standard LSM Training
```bash
!python main.py train --window-size 5 --batch-size 8 --epochs 3 --embedding-dim 64
```

#### Advanced Reservoir Training Examples

**Hierarchical Reservoir (Multi-scale processing):**
```bash
!python main.py train --reservoir-type hierarchical \
    --reservoir-config '{"scales":[{"units":128,"sparsity":0.1,"time_constant":0.05,"frequency_range":[0.5,1.0]},{"units":96,"sparsity":0.08,"time_constant":0.1,"frequency_range":[1.0,2.0]}],"global_connectivity":0.05}' \
    --window-size 5 --batch-size 8 --epochs 3
```

**Attentive Reservoir (Self-attention):**
```bash
!python main.py train --reservoir-type attentive \
    --reservoir-config '{"units":256,"num_heads":4,"sparsity":0.1,"attention_dim":64}' \
    --window-size 5 --batch-size 8 --epochs 3
```

**Echo State Reservoir (Classical ESN):**
```bash
!python main.py train --reservoir-type echo_state \
    --reservoir-config '{"units":256,"spectral_radius":0.9,"sparsity":0.1,"input_scaling":1.0}' \
    --window-size 5 --batch-size 8 --epochs 3
```

**Deep Reservoir (Multi-layer):**
```bash
!python main.py train --reservoir-type deep \
    --reservoir-config '{"layer_configs":[{"units":256,"sparsity":0.1,"frequency":1.0,"amplitude":1.0,"decay":0.1},{"units":128,"sparsity":0.08,"frequency":1.5,"amplitude":0.8,"decay":0.15}],"use_skip_connections":true}' \
    --window-size 5 --batch-size 8 --epochs 3
```

## Common Colab Issues & Solutions

### 1. CUDA/TensorFlow Warnings
The warnings you see like:
```
E0000 00:00:1754513843.578751    2437 cuda_dnn.cc:8579] Unable to register cuDNN factory...
```
These are **completely normal** in Colab and don't affect functionality. They're just TensorFlow initialization messages.

### 2. Memory Management
For large datasets in Colab, use smaller batch sizes:
```bash
!python main.py train --window-size 3 --batch-size 4 --epochs 2 --embedding-dim 32
```

### 3. Runtime Selection
- Use **GPU runtime** for faster training (Runtime → Change runtime type → GPU)
- The system automatically detects and uses available GPUs

### 4. File Persistence
Models and results are saved locally. To persist across sessions:
```python
# Download trained models
from google.colab import files
files.download('trained_model.pkl')
files.download('tokenizer.pkl')
```

## Testing Advanced Reservoirs
```bash
# Test all 4 advanced reservoir types
!python test_advanced_reservoirs.py
```

## Dataset Information
```bash
# Get info about the dialogue dataset
!python main.py data-info
```

## Troubleshooting

### Issue: "No such file or directory"
**Solution:** Make sure you're in the correct directory after cloning

### Issue: "ImportError: No module named..."
**Solution:** Run the pip install command again

### Issue: Memory errors
**Solution:** Use smaller parameters:
- Reduce `--batch-size` (try 2 or 4)
- Reduce `--window-size` (try 3)
- Reduce `--embedding-dim` (try 32)

## Complete Colab Example
```python
# Cell 1: Setup
!git clone <your-repo-url>
%cd your-repo-name
!pip install tensorflow>=2.10 numpy pandas scikit-learn requests

# Cell 2: Test advanced reservoirs
!python test_advanced_reservoirs.py

# Cell 3: Train with hierarchical reservoir
!python main.py train --reservoir-type hierarchical \
    --reservoir-config '{"scales":[{"units":64,"sparsity":0.1,"time_constant":0.05,"frequency_range":[0.5,1.0]}],"global_connectivity":0.05}' \
    --window-size 5 --batch-size 4 --epochs 2 --embedding-dim 32

# Cell 4: Download results
from google.colab import files
files.download('trained_model.pkl')
```

## Expected Output
When training starts successfully, you'll see:
```
No GPUs found. Using CPU.  # (or GPU messages if using GPU runtime)
Starting LSM training with parameters:
  Window size: 5
  Batch size: 8
  Epochs: 3
  Reservoir type: hierarchical
Loading data...
Parsed XXXXX dialogue sequences
Training data shape: X=(XXXXX, 5, 64), y=(XXXXX, 64)
```

The CUDA warnings at the beginning are normal and can be ignored!