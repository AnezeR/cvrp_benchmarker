# CVRP Benchmarker: an automated HPC-ready benchmarker for cvrp solvers

*CVRP Benchmarker* allows you to easily benchmark your cvrp solvers,
perform hyperparameter optimizaiton and manage cvrp problem instances 
both locally and in highly parallel distributed settings, such as HPC.

## Installation

Install cvrp-benchmarker from source
```bash
git clone https://github.com/AnezeR/cvrp_benchmarker
cd cvrp_benchmarker
python -m pip install .
```

## Example use

The example is provided for [HGS-CVRP](https://github.com/vidalt/HGS-CVRP).
To use it, install [PyHygese](https://github.com/chkwon/PyHygese),
which is a python wrapper for this solver.

```bash
python -m pip install hygese
```

```python
import cvrp_benchmarker as bench
import numpy
import hygese as hgs

class HygeseRunner(bench.Runner):
    @classmethod
    @property
    def hyperparameters(cls):
        # Initialize hyperparameters
        return [
            bench.HyperParameter('iter_num', (1000, 10_000)),
            bench.HyperParameter('mu', (5, 200)),
            bench.HyperParameter('lambda_', (1, 200)),
            bench.HyperParameter('el', (0., 1.)),
            bench.HyperParameter('nc', (0., 0.25), log=True),
            bench.HyperParameter('h', (0., 1.))
        ]
    
    @classmethod
    def run(cls, problem, target_time, hyperparameters):
        data = dict()

        # Adapt problem statement to your algorithm's input
        data['distance_matrix'] = problem.edge_weights
        data['demands'] = problem.demands
        data['vehicle_capacity'] = problem.capacity
        data['service_times'] = numpy.zeros(len(data['demands']))

        # Adapt hyperparameters as well
        ap = hgs.AlgorithmParameters(
            mu=hyperparameters['mu'],
            lambda_=hyperparameters['lambda_'],
            nbElite=int(hyperparameters['el'] * hyperparameters['mu']),
            nbClose=int(hyperparameters['nc'] * hyperparameters['mu']),
            nbGranular=int(hyperparameters['h'] * hyperparameters['mu']),
            nbIter=hyperparameters['iter_num'],
            # Add the time limit to parameters
            timeLimit=target_time,
        )
        
        # Solve the problem
        solver = hgs.Solver(parameters=ap, verbose=False)
        result = solver.solve_cvrp(data)

        # Return calculation results
        return result.cost


# Create a benchmarker class.
benchmarker = bench.Benchmarker(runners=[HygeseRunner], target_time=1)

# Optimize hyperparameters
benchmarker.tune_parameters(n_trials=5)

# Benchmark your runner
bench_results = benchmarker.benchmark(n_runs=4)
```
