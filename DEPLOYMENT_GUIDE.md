# LSM Production Deployment Guide

This comprehensive guide covers deploying the Sparse Sine-Activated Liquid State Machine (LSM) system in production environments.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [System Requirements](#system-requirements)
3. [Environment Setup](#environment-setup)
4. [Model Training and Validation](#model-training-and-validation)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Performance Optimization](#performance-optimization)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance and Updates](#maintenance-and-updates)

## Pre-Deployment Checklist

### ✅ Production Readiness Validation

Before deploying to production, run the comprehensive production readiness validation:

```bash
# Run enhanced production validation
python production_validation.py --output-dir production_validation

# Alternative: Run original production readiness test
python test_production_readiness.py --output-dir production_validation
```

**Requirements for Production Deployment:**
- [ ] Overall validation status: `PRODUCTION_READY` or `NEEDS_MINOR_FIXES`
- [ ] Pass rate: ≥ 80%
- [ ] All critical validations passing:
  - [ ] Environment validation (dependencies installed)
  - [ ] Training pipeline functional
  - [ ] Inference pipeline operational
  - [ ] Model management working
  - [ ] System resources adequate
  - [ ] Performance benchmarks met

### ✅ Environment Prerequisites

**Critical Dependencies:**
- [ ] Python 3.11+ installed and accessible
- [ ] TensorFlow 2.14+ with proper DLL support (Windows)
- [ ] scikit-learn for tokenization features
- [ ] All project modules importable
- [ ] Unicode support configured (Windows CMD/PowerShell)

**Windows-Specific Requirements:**
- [ ] Visual C++ Redistributable installed
- [ ] TensorFlow GPU support (optional but recommended)
- [ ] PowerShell execution policy configured
- [ ] Console encoding set to UTF-8

### ✅ Model Quality Validation

- [ ] Model achieves acceptable accuracy on validation data
- [ ] Training converges properly (no overfitting/underfitting)
- [ ] Model predictions are consistent and meaningful
- [ ] Tokenizer vocabulary covers expected input domain

### ✅ Infrastructure Readiness

- [ ] Production servers meet system requirements
- [ ] Network connectivity and security configured
- [ ] Monitoring and logging infrastructure in place
- [ ] Backup and disaster recovery procedures established

## System Requirements

### Minimum Requirements

**Hardware:**
- CPU: 4+ cores, 2.5+ GHz
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB free space for models and logs
- Network: Stable internet connection for dependencies

**Software:**
- Python 3.11+
- Operating System: Linux (Ubuntu 20.04+), Windows 10+, or macOS 10.15+

### Recommended Production Requirements

**Hardware:**
- CPU: 8+ cores, 3.0+ GHz (Intel Xeon or AMD EPYC)
- RAM: 32GB+ for large models and concurrent requests
- Storage: SSD with 50GB+ free space
- GPU: Optional but recommended for training (NVIDIA with CUDA support)

**Software:**
- Python 3.11+
- Docker (for containerized deployment)
- Load balancer (nginx, HAProxy)
- Process manager (systemd, supervisor)

### Performance Benchmarks

Based on production readiness testing:

| Metric | Target | Acceptable |
|--------|--------|------------|
| Model Load Time | < 10s | < 30s |
| Single Prediction | < 0.5s | < 1.0s |
| Batch Processing | < 0.05s/item | < 0.1s/item |
| Memory Usage | < 1GB | < 2GB |
| Concurrent Requests | 10+ | 5+ |

## Environment Setup

### 1. Python Environment

Create an isolated Python environment:

```bash
# Using venv
python -m venv lsm_production
source lsm_production/bin/activate  # Linux/macOS
# or
lsm_production\Scripts\activate  # Windows

# Using conda
conda create -n lsm_production python=3.11
conda activate lsm_production
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install missing production dependencies
pip install scikit-learn psutil gunicorn uvicorn fastapi

# For monitoring (optional)
pip install prometheus-client grafana-api

# Windows-specific: Fix TensorFlow DLL issues
pip install --upgrade tensorflow
# If issues persist, try CPU-only version:
# pip install tensorflow-cpu
```

**Windows TensorFlow Troubleshooting:**
If you encounter TensorFlow DLL errors:

1. Install Visual C++ Redistributable:
   - Download from Microsoft official site
   - Install both x64 and x86 versions

2. Use CPU-only TensorFlow for stability:
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-cpu
   ```

3. Set environment variables:
   ```cmd
   set TF_CPP_MIN_LOG_LEVEL=2
   set TF_ENABLE_ONEDNN_OPTS=0
   ```

### 3. Verify Installation

```bash
# Test core dependencies
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"

# Test project modules
python -c "from data_loader import DialogueTokenizer; print('DialogueTokenizer imported successfully')"
python -c "from model_config import ModelConfiguration; print('ModelConfiguration imported successfully')"
python -c "from src.lsm.management.model_manager import ModelManager; print('ModelManager imported successfully')"

# Run comprehensive validation
python production_validation.py

# Test basic functionality (if validation passes)
python main.py data-info --window-size 5
```

**Windows Console Setup:**
For proper Unicode support on Windows:

```cmd
# Set console to UTF-8
chcp 65001

# Or use PowerShell with UTF-8
powershell -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8"
```

### 4. Environment Variables

Create a `.env` file for production configuration:

```bash
# .env file
LSM_MODEL_PATH=/path/to/production/models
LSM_LOG_LEVEL=INFO
LSM_LOG_DIR=/var/log/lsm
LSM_CACHE_SIZE=1000
LSM_MAX_BATCH_SIZE=32
LSM_MEMORY_THRESHOLD_MB=1024
LSM_ENABLE_GPU=true
LSM_WORKERS=4
```

## Model Training and Validation

### 1. Training for Production

Train your model with production-appropriate parameters:

```bash
# Train with optimized settings
python main.py train \
    --window-size 10 \
    --embedding-dim 128 \
    --epochs 50 \
    --batch-size 32 \
    --reservoir-type standard \
    --output-dir models/production_$(date +%Y%m%d_%H%M%S)
```

### 2. Model Validation

Validate the trained model:

```bash
# Evaluate model performance
python main.py evaluate \
    --model-path models/production_20250108_120000 \
    --window-size 10 \
    --embedding-dim 128

# Test inference functionality
python inference.py \
    --model-path models/production_20250108_120000 \
    --input-text "hello how are you doing today" \
    --show-confidence
```

### 3. Model Management

Use the model manager to organize production models:

```python
from src.lsm.management.model_manager import ModelManager

manager = ModelManager()

# List available models
models = manager.list_available_models()
print("Available models:", models)

# Validate model integrity
is_valid = manager.validate_model("models/production_20250108_120000")
print("Model valid:", is_valid)

# Get model information
info = manager.get_model_info("models/production_20250108_120000")
print("Model info:", info)
```

## Production Deployment

### Option 1: Direct Python Deployment

#### 1. Create Production Script

```python
# production_server.py
#!/usr/bin/env python3
"""
Production LSM inference server.
"""

import os
import sys
from inference import OptimizedLSMInference
from lsm_logging import setup_default_logging

def main():
    # Setup logging
    setup_default_logging(level="INFO")
    
    # Load model
    model_path = os.environ.get('LSM_MODEL_PATH', 'models/latest')
    inference_engine = OptimizedLSMInference(
        model_path=model_path,
        lazy_load=True,
        cache_size=int(os.environ.get('LSM_CACHE_SIZE', 1000)),
        max_batch_size=int(os.environ.get('LSM_MAX_BATCH_SIZE', 32))
    )
    
    print(f"🚀 LSM Production Server Started")
    print(f"📁 Model: {model_path}")
    print(f"🎯 Ready for inference requests")
    
    # Start interactive session or API server
    inference_engine.interactive_session()

if __name__ == "__main__":
    main()
```

#### 2. Create Systemd Service

```ini
# /etc/systemd/system/lsm-inference.service
[Unit]
Description=LSM Inference Service
After=network.target

[Service]
Type=simple
User=lsm
Group=lsm
WorkingDirectory=/opt/lsm
Environment=LSM_MODEL_PATH=/opt/lsm/models/production
Environment=LSM_LOG_DIR=/var/log/lsm
ExecStart=/opt/lsm/venv/bin/python production_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 3. Start Service

```bash
# Enable and start service
sudo systemctl enable lsm-inference
sudo systemctl start lsm-inference

# Check status
sudo systemctl status lsm-inference

# View logs
sudo journalctl -u lsm-inference -f
```

### Option 2: Docker Deployment

#### 1. Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 lsm && chown -R lsm:lsm /app
USER lsm

# Expose port (if using web API)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "production_server.py"]
```

#### 2. Build and Run Container

```bash
# Build image
docker build -t lsm-inference:latest .

# Run container
docker run -d \
    --name lsm-production \
    -v /path/to/models:/app/models \
    -v /path/to/logs:/app/logs \
    -e LSM_MODEL_PATH=/app/models/production \
    -e LSM_LOG_LEVEL=INFO \
    --restart unless-stopped \
    lsm-inference:latest
```

#### 3. Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: '3.8'

services:
  lsm-inference:
    build: .
    container_name: lsm-production
    restart: unless-stopped
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./config:/app/config:ro
    environment:
      - LSM_MODEL_PATH=/app/models/production
      - LSM_LOG_LEVEL=INFO
      - LSM_CACHE_SIZE=1000
      - LSM_MAX_BATCH_SIZE=32
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Option 3: Web API Deployment

#### 1. Create FastAPI Server

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from inference import OptimizedLSMInference

app = FastAPI(title="LSM Inference API", version="1.0.0")

# Load model at startup
model_path = os.environ.get('LSM_MODEL_PATH', 'models/latest')
inference_engine = OptimizedLSMInference(model_path)

class PredictionRequest(BaseModel):
    dialogue_sequence: List[str]
    top_k: Optional[int] = 1
    show_confidence: Optional[bool] = False

class PredictionResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None
    top_k_predictions: Optional[List[dict]] = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if request.top_k > 1:
            predictions = inference_engine.predict_top_k(
                request.dialogue_sequence, 
                k=request.top_k
            )
            return PredictionResponse(
                prediction=predictions[0][0],
                confidence=predictions[0][1],
                top_k_predictions=[
                    {"text": text, "confidence": conf} 
                    for text, conf in predictions
                ]
            )
        elif request.show_confidence:
            prediction, confidence = inference_engine.predict_with_confidence(
                request.dialogue_sequence
            )
            return PredictionResponse(
                prediction=prediction,
                confidence=confidence
            )
        else:
            prediction = inference_engine.predict_next_token(
                request.dialogue_sequence
            )
            return PredictionResponse(prediction=prediction)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/info")
async def model_info():
    return inference_engine.get_model_info()
```

#### 2. Run with Gunicorn

```bash
# Install gunicorn
pip install gunicorn uvicorn[standard]

# Run API server
gunicorn api_server:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100
```

## Monitoring and Logging

### 1. Production Logging Configuration

Configure comprehensive logging for production environments:

```python
# production_logging.py
import logging
import logging.handlers
import os
import sys
from datetime import datetime

def setup_production_logging():
    """Setup production-grade logging with enhanced error handling."""
    
    # Create logs directory
    log_dir = os.environ.get('LSM_LOG_DIR', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Production log file handler
    production_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'lsm_production.log'),
        maxBytes=50*1024*1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    production_handler.setFormatter(detailed_formatter)
    production_handler.setLevel(logging.INFO)
    
    # Error log file handler
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'lsm_errors.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setFormatter(detailed_formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Performance log handler
    perf_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'lsm_performance.log'),
        maxBytes=20*1024*1024,  # 20MB
        backupCount=5,
        encoding='utf-8'
    )
    perf_handler.setFormatter(detailed_formatter)
    perf_handler.setLevel(logging.INFO)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(production_handler)
    root_logger.addHandler(error_handler)
    
    # Configure specific loggers
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    return {
        'production_log': os.path.join(log_dir, 'lsm_production.log'),
        'error_log': os.path.join(log_dir, 'lsm_errors.log'),
        'performance_log': os.path.join(log_dir, 'lsm_performance.log')
    }

# Usage in production
if __name__ == "__main__":
    log_files = setup_production_logging()
    logger = logging.getLogger(__name__)
    logger.info("Production logging initialized")
    logger.info(f"Log files: {log_files}")
```

### 2. Production Monitoring System

Create comprehensive monitoring for production deployment:

```python
# production_monitoring.py
import psutil
import time
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List
import threading
import queue

class ProductionMonitor:
    """Comprehensive production monitoring system."""
    
    def __init__(self, log_file='monitoring.log', alert_thresholds=None):
        self.log_file = log_file
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'prediction_time_ms': 1000.0
        }
        self.metrics_queue = queue.Queue()
        self.alerts_queue = queue.Queue()
        self.logger = logging.getLogger('production_monitor')
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('.')
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_percent': memory.percent,
                    'disk_total_gb': disk.total / (1024**3),
                    'disk_free_gb': disk.free / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100
                },
                'process': {
                    'memory_rss_mb': process_memory.rss / (1024**2),
                    'memory_vms_mb': process_memory.vms / (1024**2),
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads()
                }
            }
            
            # Add load average on Unix systems
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                metrics['system']['load_average'] = {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        try:
            # Check if models directory exists and count models
            model_count = 0
            model_size_mb = 0
            
            import glob
            model_dirs = glob.glob("models_*")
            model_count = len(model_dirs)
            
            for model_dir in model_dirs:
                try:
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            model_size_mb += os.path.getsize(os.path.join(root, file)) / (1024**2)
                except:
                    pass
            
            # Check log file sizes
            log_size_mb = 0
            if os.path.exists('logs'):
                for root, dirs, files in os.walk('logs'):
                    for file in files:
                        if file.endswith('.log'):
                            log_size_mb += os.path.getsize(os.path.join(root, file)) / (1024**2)
            
            return {
                'models': {
                    'count': model_count,
                    'total_size_mb': model_size_mb
                },
                'logs': {
                    'total_size_mb': log_size_mb
                },
                'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
            return {'error': str(e)}
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check metrics against alert thresholds."""
        alerts = []
        
        try:
            system_metrics = metrics.get('system', {})
            
            # CPU alert
            if system_metrics.get('cpu_percent', 0) > self.alert_thresholds['cpu_percent']:
                alerts.append({
                    'type': 'cpu_high',
                    'severity': 'warning',
                    'message': f"High CPU usage: {system_metrics['cpu_percent']:.1f}%",
                    'threshold': self.alert_thresholds['cpu_percent'],
                    'current_value': system_metrics['cpu_percent']
                })
            
            # Memory alert
            if system_metrics.get('memory_percent', 0) > self.alert_thresholds['memory_percent']:
                alerts.append({
                    'type': 'memory_high',
                    'severity': 'critical',
                    'message': f"High memory usage: {system_metrics['memory_percent']:.1f}%",
                    'threshold': self.alert_thresholds['memory_percent'],
                    'current_value': system_metrics['memory_percent']
                })
            
            # Disk alert
            if system_metrics.get('disk_percent', 0) > self.alert_thresholds['disk_percent']:
                alerts.append({
                    'type': 'disk_high',
                    'severity': 'critical',
                    'message': f"High disk usage: {system_metrics['disk_percent']:.1f}%",
                    'threshold': self.alert_thresholds['disk_percent'],
                    'current_value': system_metrics['disk_percent']
                })
            
            # Low memory available alert
            if system_metrics.get('memory_available_gb', 0) < 1.0:
                alerts.append({
                    'type': 'memory_low',
                    'severity': 'critical',
                    'message': f"Low available memory: {system_metrics['memory_available_gb']:.1f}GB",
                    'threshold': 1.0,
                    'current_value': system_metrics['memory_available_gb']
                })
            
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
        
        return alerts
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to file and queue."""
        try:
            # Log to file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics) + '\n')
            
            # Add to queue for real-time processing
            self.metrics_queue.put(metrics)
            
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
    
    def start_monitoring(self, interval_seconds=60):
        """Start continuous monitoring."""
        self.start_time = time.time()
        self.logger.info(f"Starting production monitoring (interval: {interval_seconds}s)")
        
        try:
            while True:
                # Collect metrics
                system_metrics = self.collect_system_metrics()
                app_metrics = self.collect_application_metrics()
                
                combined_metrics = {
                    **system_metrics,
                    'application': app_metrics
                }
                
                # Check for alerts
                alerts = self.check_alerts(combined_metrics)
                if alerts:
                    combined_metrics['alerts'] = alerts
                    for alert in alerts:
                        self.logger.warning(f"ALERT: {alert['message']}")
                
                # Log metrics
                self.log_metrics(combined_metrics)
                
                # Console output
                if 'system' in combined_metrics:
                    sys_metrics = combined_metrics['system']
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"CPU: {sys_metrics.get('cpu_percent', 0):.1f}% | "
                          f"Memory: {sys_metrics.get('memory_percent', 0):.1f}% | "
                          f"Disk: {sys_metrics.get('disk_percent', 0):.1f}% | "
                          f"Alerts: {len(alerts)}")
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")

# Usage example
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Custom alert thresholds
    thresholds = {
        'cpu_percent': 75.0,
        'memory_percent': 80.0,
        'disk_percent': 85.0,
        'prediction_time_ms': 500.0
    }
    
    # Start monitoring
    monitor = ProductionMonitor('production_metrics.log', thresholds)
    monitor.start_monitoring(interval_seconds=30)
```

### 3. Application Metrics

Add metrics to your inference code:

```python
# metrics.py
import time
import json
from collections import defaultdict, deque
from threading import Lock

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.timers = defaultdict(deque)
        self.lock = Lock()
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value
    
    def record_time(self, name: str, duration: float):
        """Record a timing metric."""
        with self.lock:
            self.timers[name].append(duration)
            # Keep only last 1000 measurements
            if len(self.timers[name]) > 1000:
                self.timers[name].popleft()
    
    def get_stats(self):
        """Get current statistics."""
        with self.lock:
            stats = {
                'counters': dict(self.counters),
                'timers': {}
            }
            
            for name, times in self.timers.items():
                if times:
                    stats['timers'][name] = {
                        'count': len(times),
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times)
                    }
            
            return stats

# Global metrics instance
metrics = MetricsCollector()
```

### 4. Health Checks

Implement comprehensive health checks:

```python
# health_check.py
import os
import time
import psutil
from inference import OptimizedLSMInference

class HealthChecker:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.inference_engine = None
        
    def check_model_loading(self) -> dict:
        """Check if model can be loaded."""
        try:
            start_time = time.time()
            self.inference_engine = OptimizedLSMInference(self.model_path)
            load_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'load_time_seconds': load_time,
                'model_path': self.model_path
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_path': self.model_path
            }
    
    def check_inference(self) -> dict:
        """Check if inference works."""
        if not self.inference_engine:
            return {'status': 'unhealthy', 'error': 'Model not loaded'}
        
        try:
            test_sequence = ["hello", "how", "are", "you", "today"]
            start_time = time.time()
            prediction = self.inference_engine.predict_next_token(test_sequence)
            inference_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'inference_time_seconds': inference_time,
                'sample_prediction': prediction
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def check_system_resources(self) -> dict:
        """Check system resource usage."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'status': 'healthy',
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / 1024**3,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / 1024**3,
                'cpu_percent': psutil.cpu_percent(interval=1)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def full_health_check(self) -> dict:
        """Perform comprehensive health check."""
        checks = {
            'model_loading': self.check_model_loading(),
            'inference': self.check_inference(),
            'system_resources': self.check_system_resources()
        }
        
        # Overall status
        all_healthy = all(
            check.get('status') == 'healthy' 
            for check in checks.values()
        )
        
        return {
            'overall_status': 'healthy' if all_healthy else 'unhealthy',
            'timestamp': time.time(),
            'checks': checks
        }
```

## Performance Optimization

### 1. Model Optimization

```python
# Optimize model loading
inference_engine = OptimizedLSMInference(
    model_path=model_path,
    lazy_load=True,          # Load components on demand
    cache_size=2000,         # Increase cache size
    max_batch_size=64        # Optimize batch size
)

# Enable GPU if available
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### 2. Memory Management

```python
# Configure memory limits
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Implement memory monitoring
def monitor_memory_usage():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 2048:  # 2GB threshold
        # Trigger garbage collection
        import gc
        gc.collect()
        
        # Clear caches if needed
        if hasattr(inference_engine, 'clear_caches'):
            inference_engine.clear_caches()
```

### 3. Caching Strategy

```python
# Implement Redis caching for distributed systems
import redis
import json
import hashlib

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.ttl = 3600  # 1 hour TTL
    
    def get_cache_key(self, sequence):
        """Generate cache key for sequence."""
        sequence_str = json.dumps(sequence, sort_keys=True)
        return hashlib.md5(sequence_str.encode()).hexdigest()
    
    def get_prediction(self, sequence):
        """Get cached prediction."""
        key = self.get_cache_key(sequence)
        cached = self.redis_client.get(key)
        return json.loads(cached) if cached else None
    
    def cache_prediction(self, sequence, prediction, confidence=None):
        """Cache prediction result."""
        key = self.get_cache_key(sequence)
        value = {'prediction': prediction, 'confidence': confidence}
        self.redis_client.setex(key, self.ttl, json.dumps(value))
```

## Security Considerations

### 1. Input Validation

```python
# Implement strict input validation
def validate_input_sequence(sequence):
    """Validate input sequence for security."""
    
    # Check type
    if not isinstance(sequence, list):
        raise ValueError("Input must be a list")
    
    # Check length
    if len(sequence) > 100:  # Prevent DoS
        raise ValueError("Input sequence too long")
    
    # Check content
    for item in sequence:
        if not isinstance(item, str):
            raise ValueError("All items must be strings")
        
        if len(item) > 1000:  # Prevent memory exhaustion
            raise ValueError("Individual items too long")
        
        # Check for malicious patterns
        if any(pattern in item.lower() for pattern in ['<script>', 'javascript:', 'eval(']):
            raise ValueError("Potentially malicious input detected")
    
    return True
```

### 2. Rate Limiting

```python
# Implement rate limiting
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id):
        """Check if request is allowed."""
        now = time.time()
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True
```

### 3. Access Control

```python
# Implement API key authentication
import hashlib
import hmac

class APIKeyAuth:
    def __init__(self, valid_keys):
        self.valid_keys = set(valid_keys)
    
    def validate_key(self, api_key):
        """Validate API key."""
        return api_key in self.valid_keys
    
    def generate_signature(self, payload, secret):
        """Generate request signature."""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
```

## Troubleshooting

### Production Validation Issues

#### 1. Environment Validation Failures

**Symptoms:**
- Missing module errors (scikit-learn, tensorflow)
- Import errors for project modules
- TensorFlow DLL initialization failures

**Solutions:**
```bash
# Install missing dependencies
pip install scikit-learn tensorflow

# For Windows TensorFlow DLL issues:
pip uninstall tensorflow
pip install tensorflow-cpu

# Verify installations
python -c "import sklearn, tensorflow as tf; print('Dependencies OK')"

# Check project module paths
python -c "import sys; print('\n'.join(sys.path))"
```

#### 2. Training Pipeline Failures

**Symptoms:**
- Training process crashes immediately
- TensorFlow runtime errors
- Memory allocation failures

**Solutions:**
```bash
# Use mock TensorFlow for testing
python -c "import sys; sys.path.insert(0, '.'); import mock_tensorflow"

# Train with minimal parameters
python main.py train --window-size 3 --embedding-dim 16 --epochs 1

# Check system resources
python -c "
import psutil
mem = psutil.virtual_memory()
print(f'Available memory: {mem.available / 1024**3:.1f}GB')
print(f'Memory usage: {mem.percent:.1f}%')
"
```

#### 3. Unicode Encoding Issues (Windows)

**Symptoms:**
- `UnicodeEncodeError: 'charmap' codec can't encode character`
- Emoji/Unicode characters not displaying
- Console output errors

**Solutions:**
```cmd
# Set console code page to UTF-8
chcp 65001

# Set environment variables
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

# Use PowerShell instead of CMD
powershell -Command "python script.py"

# Alternative: Remove Unicode characters from output
# Edit manage_models.py to use ASCII-only characters
```

### Common Issues and Solutions

#### 4. Model Loading Failures

**Symptoms:**
- `ModelLoadError` exceptions
- Long loading times
- Memory errors during loading

**Solutions:**
```bash
# Check model integrity
python -c "
from src.lsm.management.model_manager import ModelManager
manager = ModelManager()
print(manager.validate_model('path/to/model'))
"

# Check available memory
python -c "
import psutil
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f}GB')
"

# Enable lazy loading
inference_engine = OptimizedLSMInference(model_path, lazy_load=True)
```

#### 5. Performance Issues

**Symptoms:**
- Slow prediction times
- High memory usage
- CPU/GPU utilization issues

**Solutions:**
```python
# Enable performance optimizations
inference_engine = OptimizedLSMInference(
    model_path=model_path,
    lazy_load=True,
    cache_size=2000,
    max_batch_size=64
)

# Monitor performance
stats = inference_engine.get_cache_stats()
print("Cache hit rate:", stats['prediction_cache']['hit_rate'])

# Use batch processing
predictions = inference_engine.batch_predict(sequences)
```

#### 6. Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- Out of memory errors
- System slowdown

**Solutions:**
```python
# Enable memory management
inference_engine._memory_threshold_mb = 1024  # 1GB threshold
inference_engine._gc_interval = 30  # GC every 30 seconds

# Manual cleanup
inference_engine.clear_caches()
import gc; gc.collect()

# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024**2:.1f}MB")
```

### Diagnostic Commands

```bash
# Run comprehensive production validation
python production_validation.py --output-dir diagnostics

# Check system status
python -c "
import psutil
print('CPU:', psutil.cpu_percent())
print('Memory:', psutil.virtual_memory().percent, '%')
print('Disk:', psutil.disk_usage('.').percent, '%')
"

# Test model loading (if models exist)
python -c "
from inference import OptimizedLSMInference
import glob
models = glob.glob('models_*')
if models:
    engine = OptimizedLSMInference(models[0])
    print('Model loaded successfully')
    print(engine.get_model_info())
else:
    print('No models found')
"

# Test model management
python manage_models.py list --detailed

# Check logs
# Windows:
type logs\lsm_*.log | findstr ERROR
# Linux/Mac:
tail -f logs/lsm_production.log
grep ERROR logs/lsm_production.log
```

### Production Readiness Checklist

Before deploying to production, ensure all these items are completed:

#### Environment Setup
- [ ] Python 3.11+ installed
- [ ] All dependencies installed (run `pip install -r requirements.txt`)
- [ ] TensorFlow working without DLL errors
- [ ] scikit-learn installed and importable
- [ ] Unicode encoding configured (Windows)
- [ ] Production validation passes with ≥80% success rate

#### Model Training and Validation
- [ ] At least one model successfully trained
- [ ] Model files complete and uncorrupted
- [ ] Inference pipeline functional
- [ ] Performance benchmarks meet requirements
- [ ] Memory usage within acceptable limits

#### Infrastructure
- [ ] Adequate system resources (CPU, RAM, disk)
- [ ] Monitoring and logging configured
- [ ] Backup procedures established
- [ ] Security measures implemented
- [ ] Error handling and recovery procedures tested

#### Testing
- [ ] End-to-end workflow tested
- [ ] Concurrent inference tested
- [ ] Error handling validated
- [ ] Performance under load verified
- [ ] Backward compatibility confirmed

## Maintenance and Updates

### 1. Model Updates

```bash
# Backup current model
cp -r models/production models/production_backup_$(date +%Y%m%d)

# Deploy new model
cp -r models/new_model models/production

# Restart service
sudo systemctl restart lsm-inference

# Verify deployment
curl http://localhost:8000/health
```

### 2. System Updates

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Run tests after updates
python test_production_readiness.py

# Update system packages
sudo apt update && sudo apt upgrade
```

### 3. Backup Strategy

```bash
#!/bin/bash
# backup_script.sh

BACKUP_DIR="/backups/lsm/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup models
cp -r models/ "$BACKUP_DIR/models/"

# Backup configuration
cp -r config/ "$BACKUP_DIR/config/"

# Backup logs (last 7 days)
find logs/ -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/" \;

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
```

### 4. Monitoring Alerts

Set up alerts for critical metrics:

```python
# alerts.py
import smtplib
from email.mime.text import MIMEText

class AlertManager:
    def __init__(self, smtp_server, email_from, email_to):
        self.smtp_server = smtp_server
        self.email_from = email_from
        self.email_to = email_to
    
    def send_alert(self, subject, message):
        """Send email alert."""
        msg = MIMEText(message)
        msg['Subject'] = f"LSM Alert: {subject}"
        msg['From'] = self.email_from
        msg['To'] = self.email_to
        
        with smtplib.SMTP(self.smtp_server) as server:
            server.send_message(msg)
    
    def check_and_alert(self, metrics):
        """Check metrics and send alerts if needed."""
        if metrics['memory_percent'] > 90:
            self.send_alert(
                "High Memory Usage",
                f"Memory usage is {metrics['memory_percent']:.1f}%"
            )
        
        if metrics['cpu_percent'] > 90:
            self.send_alert(
                "High CPU Usage",
                f"CPU usage is {metrics['cpu_percent']:.1f}%"
            )
```

## Production Deployment Recommendations

Based on comprehensive production validation testing, here are the key recommendations for successful deployment:

### Critical Success Factors

1. **Environment Stability**
   - Ensure TensorFlow is properly installed and functional
   - Install all required dependencies (scikit-learn, psutil, etc.)
   - Configure Unicode support for Windows environments
   - Test all project modules import successfully

2. **Training Pipeline Validation**
   - Successfully train at least one model before deployment
   - Verify model files are complete and uncorrupted
   - Test inference pipeline with trained models
   - Validate performance meets requirements

3. **System Resource Management**
   - Monitor CPU, memory, and disk usage continuously
   - Implement alerting for resource threshold breaches
   - Plan for peak load scenarios
   - Establish resource cleanup procedures

4. **Monitoring and Observability**
   - Implement comprehensive logging with proper encoding
   - Set up real-time monitoring dashboards
   - Configure alerting for critical system events
   - Establish log rotation and archival policies

### Deployment Phases

#### Phase 1: Environment Preparation
- [ ] Run `python production_validation.py` and achieve ≥80% pass rate
- [ ] Fix all critical validation failures
- [ ] Establish monitoring and logging infrastructure
- [ ] Create backup and recovery procedures

#### Phase 2: Model Training and Validation
- [ ] Train production models with appropriate parameters
- [ ] Validate model performance and accuracy
- [ ] Test inference pipeline thoroughly
- [ ] Benchmark performance under expected load

#### Phase 3: Production Deployment
- [ ] Deploy to staging environment first
- [ ] Conduct load testing and stress testing
- [ ] Validate all monitoring and alerting systems
- [ ] Execute production deployment with rollback plan

#### Phase 4: Post-Deployment Monitoring
- [ ] Monitor system performance continuously
- [ ] Track application metrics and user feedback
- [ ] Implement automated health checks
- [ ] Plan for regular maintenance and updates

### Windows-Specific Considerations

Due to the Windows environment challenges identified during validation:

1. **TensorFlow Installation**
   - Consider using `tensorflow-cpu` for stability
   - Install Visual C++ Redistributable packages
   - Set appropriate environment variables

2. **Unicode Handling**
   - Configure console encoding to UTF-8
   - Use PowerShell instead of CMD when possible
   - Test all output formatting thoroughly

3. **Performance Optimization**
   - Monitor memory usage closely on Windows
   - Consider containerization for consistency
   - Implement proper resource cleanup

### Continuous Improvement

1. **Regular Validation**
   - Run production validation monthly
   - Update validation criteria based on operational experience
   - Maintain validation test coverage

2. **Performance Monitoring**
   - Track key performance indicators
   - Establish performance baselines
   - Implement automated performance regression detection

3. **System Updates**
   - Plan regular dependency updates
   - Test updates in staging environment first
   - Maintain rollback procedures for all changes

## Conclusion

This deployment guide provides comprehensive instructions for deploying the LSM system in production environments, with specific attention to the challenges identified through production validation testing.

### Key Success Factors:

1. **Thorough Validation** - Always run comprehensive production validation before deployment
2. **Environment Stability** - Ensure all dependencies are properly installed and configured
3. **Continuous Monitoring** - Implement robust monitoring and alerting systems
4. **Proper Planning** - Follow phased deployment approach with rollback capabilities
5. **Regular Maintenance** - Establish procedures for ongoing system maintenance and updates

### Critical Resources:

- **Production Validation**: Run `python production_validation.py` before any deployment
- **Monitoring**: Use the production monitoring system for continuous oversight
- **Troubleshooting**: Refer to the comprehensive troubleshooting section for common issues
- **Performance**: Monitor system resources and application performance continuously

For additional support and detailed technical information, refer to:
- [API Documentation](API_DOCUMENTATION.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
- [Performance Optimization Summary](PERFORMANCE_OPTIMIZATION_SUMMARY.md)
- [Comprehensive Test Summary](COMPREHENSIVE_TEST_SUMMARY.md)

---

**Last Updated:** August 7, 2025  
**Version:** 2.0  
**Maintainer:** LSM Development Team  
**Production Validation Status:** Requires environment fixes for full readiness