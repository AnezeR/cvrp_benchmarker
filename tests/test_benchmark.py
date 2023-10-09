import os
from tempfile import TemporaryDirectory
from typing import Optional, Union, Sequence
from math import isclose
import time

from optuna.study import Study

import cvrp_benchmarker as bench

from .sisr_runner import SISR_Runner
import numpy


def test_instance_downloading():
    with TemporaryDirectory() as tempdir:
        mock_problem_name_1 = 'X-n134-k13'
        mock_problem_name_2 = 'X-n125-k30'
        bench.Benchmarker(
            runners = [],
            problem_names=[mock_problem_name_1, mock_problem_name_2],
            problems_dir=tempdir
        )

        assert os.path.exists(os.path.join(tempdir, f'{mock_problem_name_1}.ins'))
        assert os.path.exists(os.path.join(tempdir, f'{mock_problem_name_1}.sol'))
        
        assert os.path.exists(os.path.join(tempdir, f'{mock_problem_name_2}.ins'))
        assert os.path.exists(os.path.join(tempdir, f'{mock_problem_name_2}.sol'))
        
def test_studies_init():
    class MockRunner(bench.Runner):
        @classmethod
        def run(cls):
            pass
    
    with TemporaryDirectory() as tempdir:
        benchmarker = bench.Benchmarker(
            runners=[SISR_Runner, MockRunner],
            problem_names=[],
            study_storage=f'sqlite:///{tempdir}/cvrp_parameter_tuning.db',
        )
        
        assert isinstance(benchmarker.studies[0][1], Study)
        assert benchmarker.studies[1][1] is None

def test_parallelization_capabilities():
    mock_problem_names = ['X-n134-k13']
    wait_time = 5

    # Testing for
    # worker 1 : runner1-task1 -> runner3-task1 -> runner2-task2
    # worker 2 : runner2-task1 -> runner1-task2 -> runner3-task2
    class MockRunner1(bench.Runner):
        @classmethod
        @property
        def name(cls) -> str:
            return 'mock_runner1'
        @classmethod
        @property
        def hyperparameters(cls):
            return [bench.HyperParameter('a', (2, 4))]
        @classmethod
        def run(cls, *args, **kwargs):
            time.sleep(wait_time)
            return 42.
    class MockRunner2(MockRunner1):
        @classmethod
        @property
        def name(cls) -> str:
            return 'mock_runner2'
    class MockRunner3(MockRunner1):
        @classmethod
        @property
        def name(cls) -> str:
            return 'mock_runner3'
    class MockRunner4(MockRunner1):
        @classmethod
        @property
        def hyperparameters(cls):
            pass # Checking if this one is not trained, so the training pattern does not include this runner
        @classmethod
        @property
        def name(cls) -> str:
            return 'mock_runner4'
    with TemporaryDirectory() as tempdir:
        benchmarker = bench.Benchmarker(
            runners = [MockRunner1, MockRunner2, MockRunner3, MockRunner4],
            problem_names=mock_problem_names,
            problems_dir=tempdir,
            study_storage=f'sqlite:///{tempdir}/cvrp_parameter_tuning.db',
            target_time=5, # 5s runtime is chosen to be bigger than overhead
        )
        start_bench = time.time()
        benchmarker.tune_parameters(n_trials=2, cluster_pool=bench.ClusterPool(n_clusters=2, workers_per_cluster=2))
        end_bench = time.time()
        assert benchmarker.studies[0][1].best_trial is not None
        assert benchmarker.studies[1][1].best_trial is not None
        assert benchmarker.studies[2][1].best_trial is not None
        assert isclose(start_bench, end_bench - wait_time * 3, abs_tol=3) # aprox. 3s (~2.8) is lost at execution as overhead.


def test_benchmarking():
    mock_problem_names = ['X-n134-k13']

    class MockRunner1(bench.Runner):
        @classmethod
        @property
        def name(cls) -> str:
            return 'mock_runner1'
        @classmethod
        @property
        def hyperparameters(cls):
            return [bench.HyperParameter('a', (2, 4))]
        @classmethod
        def run(cls, *args, **kwargs):
            return 42.
    class MockRunner2(MockRunner1):
        @classmethod
        @property
        def name(cls) -> str:
            return 'mock_runner2'
    class MockRunner3(MockRunner1):
        @classmethod
        @property
        def hyperparameters(cls):
            pass # Checking if this one is not trained, so the training pattern does not include this runner
        @classmethod
        @property
        def name(cls) -> str:
            return 'mock_runner3'
        @classmethod
        def run(cls, *args, **kwargs):
            return 12
    with TemporaryDirectory() as tempdir:
        benchmarker = bench.Benchmarker(
            runners = [MockRunner1, MockRunner2, MockRunner3],
            problem_names=mock_problem_names,
            problems_dir=tempdir,
            study_storage=f'sqlite:///{tempdir}/cvrp_parameter_tuning.db',
            target_time=5, # 5s runtime is chosen to be bigger than overhead
        )
        benchmarker.benchmark(n_runs=4, cluster_pool=bench.ClusterPool(n_clusters=4))