import argparse
from experiments.run_benchmark import run_experiment
from experiments.dataset_factory import get_benchmarks

def main():
    parser = argparse.ArgumentParser(description="CARE-EDGE Benchmark Runner")
    parser.add_argument("--benchmark", type=str, default="BDD100K", choices=["BDD100K", "METR-LA", "UrbanSound8K"],
                        help="Benchmark to run")
    parser.add_argument("--shift", type=str, default="medium", choices=["mild", "medium", "severe"],
                        help="Shift severity level")
    parser.add_argument("--seed", type=str, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    benchmarks = get_benchmarks()
    config = benchmarks[args.benchmark]
    
    print(f"Starting experiment: Benchmark={args.benchmark}, Shift={args.shift}, Seed={args.seed}")
    results = run_experiment(args.benchmark, config, shift_mode=args.shift, seed=int(args.seed))
    
    print("\n--- Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
