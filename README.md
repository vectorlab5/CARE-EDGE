# CARE-EDGE: Conformal Risk-Controlled Edge–Cloud Inference Routing

CARE-EDGE is a deployment-time framework for smart-city cyber-physical systems that optimizes the trade-off between edge device efficiency and cloud model accuracy. It uses Conformal Risk Control (CRC) to ensure that served predictions meet a user-specified risk budget while minimizing unnecessary cloud escalations.

## Key Features

- **Selective Routing**: Dynamically decides whether to process an input on the edge or escalate to the cloud based on real-time confidence scores.
- **Risk Control**: Implements Conformal Risk Control (CRC) to provide finite-sample guarantees on the served risk level ($\alpha$).
- **Drift Adaptation**: Uses an online adversarial probe to detect deployment-time distribution shift and automatically tighten the routing gate.
- **Authenticated Provenance**: Generates HMAC-based audit tags and Merkle roots for tamper-evident logging of inference decisions.
- **Multi-Modal Support**: Optimized backbones for Vision (MobileNetV3), Traffic (Transformer), and Acoustic (AST) sensing tasks.

## Project Structure

```text
code/
├── care_edge/              # Core framework library
│   ├── engine.py           # Main routing engine (Algorithm 1)
│   ├── models/             # Specialized backbones and scoring heads
│   ├── modules/            # CRC updater, probe, and provenance modules
│   └── utils/              # CUSUM and sliding window utilities
├── experiments/            # Evaluation harness
│   ├── dataset_factory.py  # Data loaders and shift simulators
│   └── run_benchmark.py    # Main script for running benchmarks
└── main.py                 # CLI entry point
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/care-edge.git
cd care-edge/code

# Install dependencies
pip install torch torchvision numpy
```

## Usage

### Running Benchmarks
To evaluate CARE-EDGE across the standard vision, traffic, and audio benchmarks:

```bash
python3 experiments/run_benchmark.py
```

### Basic Integration
```python
from care_edge.engine import CareEdgeEngine
from care_edge.models.backbones import VisionBackbone

# Initialize models
f_e = VisionBackbone(num_classes=4)
f_c = VisionBackbone(num_classes=4)

# Configuration
config = {
    "alpha": 0.1,    # Risk budget
    "W": 2048,       # Calibration window size
    "r": 0.01,       # Probe rate
    "epsilon": 0.03, # Perturbation budget
    "N": 1000        # Provenance batch size
}

# Initialize engine
engine = CareEdgeEngine(f_e, f_c, config)

# Process a stream of inputs
for x in data_stream:
    result = engine.step(x)
    print(f"Prediction: {result['prediction']}, Route: {result['route']}")
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
