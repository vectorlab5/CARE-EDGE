# CARE-EDGE: Conformal Risk-Controlled Edge–Cloud Inference Routing

CARE-EDGE is a deployment-time framework for smart-city cyber-physical systems that optimizes the trade-off between edge device efficiency and cloud model accuracy. It uses Conformal Risk Control (CRC) to ensure that served predictions meet a user-specified risk budget while minimizing unnecessary cloud escalations.

## Key Features

- **Selective Routing**: Dynamically decides whether to process an input on the edge or escalate to the cloud based on real-time confidence scores.
- **Risk Control**: Implements Conformal Risk Control (CRC) to provide finite-sample guarantees on the served risk level ($\alpha$).
- **Advanced Scoring**: Supports classification (-log p), entropy, and **Conformalized Quantile Regression** for time-series anomaly detection.
- **Drift Adaptation**: Uses an online adversarial probe (FGSM, **30-step PGD**) and natural corruption diagnostics to automatically tighten the routing gate.
- **Authenticated Provenance**: Generates HMAC-SHA256 audit tags and Merkle roots for tamper-evident logging, following exact architectural specifications for state commitment.
- **Comprehensive Baselines**: Includes implementations for `Always-edge`, `Always-cloud`, `Entropy-threshold`, `DRL-offload`, and `Bayes-variance` (ensemble).
- **Multi-Modal Support**: Optimized backbones for Vision (MobileNetV3), Traffic (Transformer Encoder with Positional Embeddings), and Acoustic (Audio Spectrogram Transformer) sensing.

## Project Structure

```text
code/
├── care_edge/              # Core framework library
│   ├── engine.py           # Main routing engine (Algorithm 1)
│   ├── baselines.py        # Implementation of 5 comparison baselines
│   ├── models/             
│   │   ├── backbones.py    # MobileNetV3, Transformer, AST models
│   │   └── scoring.py      # Multi-modal scoring heads and ECE metrics
│   ├── modules/            
│   │   ├── updater.py      # CRC and Quantile-based threshold logic
│   │   ├── probe.py        # PGD and natural corruption perturbations
│   │   └── provenance.py   # HMAC and Merkle audit trail construction
│   └── utils/              
│       ├── metrics.py      # Mis-coverage, cost, ECE, and reward tracking
│       └── stats.py        # CUSUM and efficient sliding windows
├── experiments/            # Evaluation harness
│   ├── dataset_factory.py  # Data loaders with shift/attack simulation
│   ├── run_benchmark.py    # Main script for running the evaluation grid
│   └── reproduce_tables.py # Reproduction script for Table 2 and ablations
└── main.py                 # CLI entry point
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/care-edge.git
cd care-edge/code

# Install dependencies
pip install torch torchvision numpy pandas
```

## Usage

### Running the Full Benchmark
To reproduce the main results and evaluation grid:

```bash
python3 experiments/run_benchmark.py
```

### Reproducing Paper Tables
To generate the summary tables and ablation results:

```bash
python3 experiments/reproduce_tables.py
```

### CLI Options
```bash
python3 main.py --benchmark BDD100K --shift medium --seed 42
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
