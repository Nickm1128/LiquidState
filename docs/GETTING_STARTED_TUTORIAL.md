# Getting Started with LSM Convenience API

Welcome to the LSM (Liquid State Machine) Convenience API! This tutorial will guide you through your first steps with the simplified, scikit-learn-like interface for creating powerful neural language models.

## What is the LSM Convenience API?

The LSM Convenience API provides a simple, familiar interface for training and using Liquid State Machine models. Instead of dealing with complex multi-component architectures, you can now create sophisticated language models with just a few lines of code.

## Installation and Setup

### Prerequisites

Make sure you have Python 3.8+ and the required dependencies:

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
from lsm import LSMGenerator, LSMClassifier, LSMRegressor
print("✅ LSM Convenience API installed successfully!")
```

## Your First LSM Model

Let's start with a simple text generation example:

```python
from lsm import LSMGenerator

# Create a generator with default settings
generator = LSMGenerator()

# Prepare some training data
conversations = [
    "User: Hello!\nAssistant: Hi there! How can I help you today?",
    "User: What's the weather like?\nAssistant: I don't have access to current weather data, but I'd be happy to help with other questions!",
    "User: Thanks!\nAssistant: You're welcome! Feel free to ask if you need anything else."
]

# Train the model (this is a minimal example - real training needs more data)
print("Training the model...")
generator.fit(conversations, epochs=10)  # Using fewer epochs for demo

# Generate a response
response = generator.generate("Hello, how are you?")
print(f"Generated response: {response}")
```

## Understanding the Basics

### The Three Main Classes

The convenience API provides three main classes:

1. **LSMGenerator**: For text generation and conversational AI
2. **LSMClassifier**: For text classification tasks
3. **LSMRegressor**: For regression and time series prediction

### Configuration Presets

Instead of manually tuning parameters, use presets:

```python
from lsm import LSMGenerator
from lsm.convenience import ConvenienceConfig

# See available presets
presets = ConvenienceConfig.list_presets()
for name, description in presets.items():
    print(f"{name}: {description}")

# Use a preset
generator = LSMGenerator(preset='fast')  # Good for experimentation
# generator = LSMGenerator(preset='balanced')  # Good balance
# generator = LSMGenerator(preset='quality')   # Best quality
```

## Tutorial 1: Text Generation

Let's build a simple chatbot:

```python
from lsm import LSMGenerator

# Step 1: Prepare training data
training_conversations = [
    "User: Hi\nAssistant: Hello! How can I help you?",
    "User: What's your name?\nAssistant: I'm an AI assistant created with LSM technology.",
    "User: Can you help me?\nAssistant: Of course! I'm here to help. What do you need?",
    "User: Tell me a joke\nAssistant: Why don't scientists trust atoms? Because they make up everything!",
    "User: That's funny\nAssistant: I'm glad you enjoyed it! Do you want to hear another one?",
    "User: Goodbye\nAssistant: Goodbye! Have a great day!"
]

# Step 2: Create and configure the generator
generator = LSMGenerator(
    preset='balanced',  # Good balance of speed and quality
    system_message_support=True  # Enable system messages
)

# Step 3: Train the model
print("Training chatbot...")
generator.fit(
    training_conversations,
    epochs=50,  # More epochs for better quality
    validation_split=0.2  # Use 20% for validation
)

# Step 4: Test the chatbot
test_prompts = [
    "Hello there!",
    "What can you do?",
    "Tell me something interesting"
]

print("\n🤖 Chatbot responses:")
for prompt in test_prompts:
    response = generator.generate(
        prompt,
        system_message="You are a friendly and helpful assistant",
        max_length=50,
        temperature=0.8  # Slightly creative responses
    )
    print(f"User: {prompt}")
    print(f"Bot: {response}\n")

# Step 5: Save the model
generator.save("my_chatbot")
print("✅ Chatbot saved to 'my_chatbot' directory")
```

### Loading and Using Saved Models

```python
# Load the saved model
loaded_generator = LSMGenerator.load("my_chatbot")

# Use it immediately
response = loaded_generator.generate("How are you today?")
print(f"Response: {response}")
```

## Tutorial 2: Text Classification

Let's build a sentiment classifier:

```python
from lsm import LSMClassifier
import numpy as np

# Step 1: Prepare classification data
texts = [
    "I love this product! It's amazing!",
    "This is the worst thing I've ever bought.",
    "It's okay, nothing special but not bad either.",
    "Absolutely fantastic! Highly recommend!",
    "Terrible quality, waste of money.",
    "Pretty good, does what it's supposed to do.",
    "Outstanding service and great quality!",
    "Not worth the price, very disappointed.",
    "Decent product, meets expectations.",
    "Excellent! Will buy again!"
]

labels = [
    "positive", "negative", "neutral",
    "positive", "negative", "neutral",
    "positive", "negative", "neutral", "positive"
]

# Step 2: Create and train classifier
classifier = LSMClassifier(
    classifier_type='random_forest',  # Use random forest for classification
    window_size=8  # Smaller window for shorter texts
)

print("Training sentiment classifier...")
classifier.fit(texts, labels, epochs=30)

# Step 3: Test the classifier
test_texts = [
    "This product is incredible!",
    "I hate this so much.",
    "It's an average product."
]

predictions = classifier.predict(test_texts)
probabilities = classifier.predict_proba(test_texts)

print("\n📊 Classification results:")
for text, pred, probs in zip(test_texts, predictions, probabilities):
    print(f"Text: '{text}'")
    print(f"Prediction: {pred}")
    print(f"Probabilities: {dict(zip(classifier.classes_, probs))}\n")

# Step 4: Evaluate performance
accuracy = classifier.score(texts, labels)
print(f"Training accuracy: {accuracy:.2f}")
```

## Tutorial 3: Time Series Regression

Let's predict the next value in a sequence:

```python
from lsm import LSMRegressor
import numpy as np

# Step 1: Create synthetic time series data
def create_time_series(n_samples=100):
    """Create a simple sine wave with noise"""
    x = np.linspace(0, 4*np.pi, n_samples)
    y = np.sin(x) + 0.1 * np.random.randn(n_samples)
    return y

# Generate data
time_series = create_time_series(200)

# Create sequences (use past 10 values to predict next value)
window_size = 10
X, y = [], []
for i in range(window_size, len(time_series)):
    X.append(time_series[i-window_size:i])
    y.append(time_series[i])

X, y = np.array(X), np.array(y)

# Step 2: Split data
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Step 3: Create and train regressor
regressor = LSMRegressor(
    reservoir_type='echo_state',  # Good for time series
    regressor_type='ridge',  # Ridge regression for stability
    window_size=window_size
)

print("Training time series regressor...")
regressor.fit(X_train, y_train, epochs=40)

# Step 4: Make predictions
predictions = regressor.predict(X_test)

# Step 5: Evaluate
r2_score = regressor.score(X_test, y_test)
mse = np.mean((predictions - y_test) ** 2)

print(f"\n📈 Time series prediction results:")
print(f"R² Score: {r2_score:.3f}")
print(f"MSE: {mse:.3f}")

# Show some predictions
print("\nSample predictions:")
for i in range(min(5, len(predictions))):
    print(f"Actual: {y_test[i]:.3f}, Predicted: {predictions[i]:.3f}")
```

## Advanced Features

### System Messages for Better Control

```python
generator = LSMGenerator(system_message_support=True)

# Train with system messages
conversations_with_system = [
    {
        "messages": ["Hello", "Hi there!", "How are you?", "I'm doing well!"],
        "system": "You are a friendly assistant"
    },
    {
        "messages": ["What's 2+2?", "2+2 equals 4", "Thanks!", "You're welcome!"],
        "system": "You are a math tutor"
    }
]

generator.fit(conversations_with_system)

# Generate with different system messages
math_response = generator.generate(
    "What's 5+3?",
    system_message="You are a math tutor. Explain your answers clearly."
)

friendly_response = generator.generate(
    "How's your day?",
    system_message="You are a friendly companion. Be warm and engaging."
)
```

### Batch Processing for Efficiency

```python
# Generate multiple responses at once
prompts = [
    "Hello!",
    "What's the weather?",
    "Tell me a fact",
    "How are you?"
]

responses = generator.batch_generate(
    prompts,
    max_length=30,
    temperature=0.7
)

for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

### Interactive Chat Sessions

```python
# Start an interactive chat (in terminal)
generator.chat(system_message="You are a helpful coding assistant")
# This opens an interactive session where you can chat with the model
```

## Best Practices

### 1. Start with Presets

```python
# For experimentation
generator = LSMGenerator(preset='fast')

# For production
generator = LSMGenerator(preset='quality')
```

### 2. Use Appropriate Data Amounts

```python
# Minimum recommended data sizes:
# - Text generation: 100+ conversations
# - Classification: 50+ examples per class
# - Regression: 100+ sequences
```

### 3. Monitor Training

```python
generator.fit(
    data,
    validation_split=0.2,  # Always use validation
    epochs=50,
    verbose=True  # Monitor progress
)
```

### 4. Handle Errors Gracefully

```python
from lsm.convenience import ConvenienceValidationError

try:
    generator = LSMGenerator(window_size=-1)  # Invalid parameter
except ConvenienceValidationError as e:
    print(f"Error: {e}")
    print(f"Suggestion: {e.suggestion}")
    print(f"Valid options: {e.valid_options}")
```

### 5. Save Your Models

```python
# Always save trained models
generator.fit(data)
generator.save("my_model_v1")

# Load when needed
generator = LSMGenerator.load("my_model_v1")
```

## Performance Tips

### Memory Management

```python
# For limited memory
generator = LSMGenerator(
    batch_size=8,  # Smaller batches
    auto_memory_management=True  # Automatic optimization
)
```

### Speed Optimization

```python
# For faster training
generator = LSMGenerator(
    preset='fast',
    epochs=20,  # Fewer epochs
    validation_split=0.1  # Less validation data
)
```

### Quality Optimization

```python
# For best quality
generator = LSMGenerator(
    preset='quality',
    epochs=100,  # More training
    validation_split=0.2,
    window_size=20  # Larger context
)
```

## Common Patterns

### Pattern 1: Quick Experimentation

```python
from lsm import LSMGenerator

# Quick setup for testing ideas
generator = LSMGenerator(preset='fast')
generator.fit(small_dataset, epochs=10)
result = generator.generate("test prompt")
```

### Pattern 2: Production Training

```python
from lsm import LSMGenerator

# Careful setup for production
generator = LSMGenerator(
    preset='quality',
    random_state=42  # Reproducibility
)

generator.fit(
    large_dataset,
    validation_split=0.2,
    epochs=100,
    batch_size=32
)

generator.save("production_model_v1")
```

### Pattern 3: Sklearn Integration

```python
from sklearn.model_selection import cross_val_score
from lsm import LSMClassifier

classifier = LSMClassifier()
scores = cross_val_score(classifier, texts, labels, cv=5)
print(f"Cross-validation scores: {scores}")
```

## Next Steps

Now that you've learned the basics:

1. **Explore Examples**: Check out the `examples/` directory for more detailed examples
2. **Read the API Documentation**: See `docs/CONVENIENCE_API_DOCUMENTATION.md` for complete API reference
3. **Try Advanced Features**: Experiment with different reservoir types and configurations
4. **Join the Community**: Share your projects and get help from other users

## Troubleshooting

### Common Issues

**Import Error**: Make sure the `src/` directory is in your Python path:
```python
import sys
sys.path.append('path/to/lsm/src')
```

**Memory Error**: Use smaller batch sizes or enable auto memory management:
```python
generator = LSMGenerator(batch_size=8, auto_memory_management=True)
```

**Poor Results**: Try more training data, more epochs, or the 'quality' preset:
```python
generator = LSMGenerator(preset='quality')
generator.fit(data, epochs=100)
```

### Getting Help

- Check the troubleshooting guide: `docs/TROUBLESHOOTING_GUIDE.md`
- Look at examples: `examples/` directory
- Enable debug logging for detailed information

Happy modeling with LSM! 🚀