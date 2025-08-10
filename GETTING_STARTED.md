# Getting Started with LSM Enhanced Pipeline

Welcome to the Liquid State Machine (LSM) project with enhanced tokenizer convenience functions! This guide will help you get started quickly.

## 🚀 Quick Start (5 minutes)

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd lsm-project

# Install the package
pip install -e .
```

### 2. Validate Installation
```bash
# Run the validation script
python validate_enhanced_pipeline.py
```

### 3. Try the Demo
```bash
# Run the enhanced tokenizer demo
python examples/enhanced_tokenizer_demo.py
```

### 4. Open Colab Notebook
- Upload `LSM_Enhanced_Pipeline_Demo.ipynb` to Google Colab
- Follow the step-by-step demonstration

## 🔤 Enhanced Tokenizer Quick Example

```python
from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
from lsm.convenience import LSMGenerator

# Create enhanced tokenizer (automatic backend detection)
tokenizer = EnhancedTokenizerWrapper(
    tokenizer='gpt2',  # or 'bert-base-uncased'
    embedding_dim=256,
    enable_caching=True
)

# Create sinusoidal embedder
embedder = tokenizer.create_configurable_sinusoidal_embedder(
    learnable_frequencies=True
)

# Use with LSM Generator
generator = LSMGenerator(
    tokenizer='gpt2',
    embedding_type='configurable_sinusoidal',
    reservoir_type='attentive',
    system_message_support=True
)

# Train on conversation data
conversations = ["User: Hello\\nAssistant: Hi there!"]
generator.fit(conversations, epochs=10)

# Generate responses
response = generator.generate("How are you?", temperature=0.8)
print(response)
```

## 📚 What's Available

### 🎯 Main Features
- **Enhanced Tokenizer**: Automatic backend detection (GPT-2, BERT, etc.)
- **Sinusoidal Embeddings**: Configurable positional encoding
- **LSM Generator**: Complete pipeline with convenience API
- **System Messages**: Context-aware response generation
- **Batch Processing**: Efficient multi-prompt generation

### 📖 Documentation
- **[Enhanced Tokenizer Guide](ENHANCED_TOKENIZER_CONVENIENCE_GUIDE.md)** - Comprehensive guide
- **[Project Structure](PROJECT_STRUCTURE.md)** - Project organization
- **[Migration Guide](MIGRATION_GUIDE.md)** - Upgrading from older versions
- **[API Documentation](docs/)** - Detailed API reference

### 🎭 Examples & Demos
- **[Colab Demo](LSM_Enhanced_Pipeline_Demo.ipynb)** - Complete pipeline demonstration
- **[Enhanced Tokenizer Demo](examples/enhanced_tokenizer_demo.py)** - Standalone demo
- **[Examples Directory](examples/)** - Various usage examples

## 🎯 Common Use Cases

### 1. Text Generation
```python
generator = LSMGenerator(tokenizer='gpt2', preset='balanced')
generator.fit(conversations)
response = generator.generate("Tell me about AI")
```

### 2. System-Aware Chatbot
```python
response = generator.generate(
    "Explain neural networks",
    system_message="You are a helpful AI teacher",
    temperature=0.7
)
```

### 3. Batch Processing
```python
prompts = ["Hello", "How are you?", "What's AI?"]
responses = generator.batch_generate(prompts, temperature=0.8)
```

### 4. Custom Tokenizer Backend
```python
tokenizer = EnhancedTokenizerWrapper(
    tokenizer='bert-base-uncased',
    embedding_dim=512,
    backend_specific_config={'do_lower_case': True}
)
```

## 🛠️ Configuration Presets

```python
from lsm.convenience.config import ConvenienceConfig

# Fast training (lower quality, faster)
fast_config = ConvenienceConfig.get_preset('fast')

# Balanced (good quality/speed tradeoff)
balanced_config = ConvenienceConfig.get_preset('balanced')

# Quality (best quality, slower)
quality_config = ConvenienceConfig.get_preset('quality')

# Customize any preset
custom_config = ConvenienceConfig.get_preset('balanced')
custom_config.update({
    'tokenizer': 'bert-base-uncased',
    'embedding_dim': 512,
    'reservoir_type': 'hierarchical'
})

generator = LSMGenerator(**custom_config)
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure package is installed
   pip install -e .
   ```

2. **Memory Issues**
   ```python
   # Use smaller batch sizes and embedding dimensions
   generator = LSMGenerator(embedding_dim=128, preset='fast')
   ```

3. **GPU Issues**
   ```python
   # Check GPU availability
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

### Getting Help
- Check the [Troubleshooting Guide](docs/TROUBLESHOOTING_GUIDE.md)
- Run the validation script: `python validate_enhanced_pipeline.py`
- Look at working examples in the `examples/` directory

## 🎉 Next Steps

1. **Explore Examples**: Check out the `examples/` directory for more use cases
2. **Read Documentation**: Dive deeper with the comprehensive guides
3. **Experiment**: Try different tokenizer backends and configurations
4. **Deploy**: Use the trained models in your applications

## 📞 Support

- **Documentation**: Check the `docs/` directory
- **Examples**: Look at working code in `examples/`
- **Validation**: Run `python validate_enhanced_pipeline.py`
- **Issues**: Check the troubleshooting guides

Happy experimenting with LSM! 🚀