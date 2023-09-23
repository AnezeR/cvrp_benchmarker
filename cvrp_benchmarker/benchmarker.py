import os
from contextlib import nullcontext

from typing import Optional, Sequence, Type
from numpy.typing import NDArray

from math import isclose

import optuna
from concurrent.futures import ThreadPoolExecutor
from dask.distributed import Client, LocalCluster
from functools import partial

from multiprocessing import Pool
import threading

import numpy
import pandas
import vrplib

from matplotlib import pyplot as plt, patches
from matplotlib.path import Path

from .problem import VrplibProblem
from .runner import Runner, CheckedRunResult, TrustedRunResult


class NotAllCustomersException(optuna.exceptions.TrialPruned):
    pass

class CapacitiesExceededException(optuna.exceptions.TrialPruned):
    pass

class WrongCostException(optuna.exceptions.TrialPruned):
    pass

class TargetTimeExceededException(optuna.exceptions.TrialPruned):
    pass

class UnknownResultFormat(optuna.exceptions.TrialPruned):
    pass

class Benchmarker:
    """Class that handles hyperparameter tuning and instance managements.
    """

    def __init__(
            self,
            runners: Sequence[Type[Runner]],
            target_time: int = 60,
            study_storage: str = 'sqlite:///cvrp_parameter_tuning.db',
            study_reset: bool = False,
            no_study_storage: bool = False,
            study_sampler: optuna.samplers.BaseSampler = optuna.samplers.NSGAIIISampler,
            problem_names: Optional[Sequence[str]] = None,
            problems_dir: Optional[str] = None,
            plots_dir: Optional[str] = None,
            plot_cost_history: bool = False,
            plot_solutions: bool = True,
            checkpoint_dir: Optional[str] = None,
    ):
        """
        :param runners: Runners to be benchmarked
        :param client: Dask client to run on
            Default: None (inititalizes a local client)
        :param target_time: Target time for benchmarking in seconds
            Default: 60
        :param study_storage: Storage for an optuna study.
            Default: 'sqlite:///cvrp_parameter_tuning.db'
        :param study_reset: Reset existing studies for runners in the storage.
            Default: False
        :param study_sampler: Sampler for the optuna study.
            Default: NSGAIIISampler
        :param problem_names: Names of problems to run. Must be present in vrplib.
            Default: None (all problems from X-series)
        :param problems_dir: Directory for storing problems.
            Default: None (Problems will be downloaded to a temporary directory)
        :param plots_dir: Directory for storing plots. No plots are stored if None.
            Default: None (no plots are stored)
        :param plot_cost_history: Plots cost history if True
            Default: False
        :param plot_solutions: Plots solutions if True
            Default: True (Only plots if plots_dir is not None)
        """
        
        self.runners = runners
        
        if problem_names is None:
            problem_names = vrplib.list_names(100, 1000, 'cvrp')
            problem_names = [
                p_name
                for p_name in problem_names if (
                    p_name.startswith('X-')  # Only leaving X-series benchmarks
                )
            ]
        self.problems = [VrplibProblem(problem_name, problems_dir) for problem_name in problem_names]
        
        self.target_time = target_time
        
        if plots_dir is not None:
            self.plots_dir = plots_dir
            if plot_solutions:
                self.solution_plots_dir = os.path.join(self.plots_dir, 'solutions')
                for runner in self.runners:
                    runner_plots_dir = os.path.join(self.solution_plots_dir, runner.name)
                    if not os.path.exists(runner_plots_dir):
                        os.mkdir(runner_plots_dir)
            if plot_cost_history:
                self.history_plots_dir = os.path.join(self.plots_dir, 'cost_hitsories')
                for runner in self.runners:
                    runner_history_dir = os.path.join(self.history_plots_dir, runner.name)
                    if not os.path.exists(runner_history_dir):
                        os.mkdir(runner_history_dir)
        
        self.studies = [
            (runner, optuna.create_study(
                study_name=runner.name,
                sampler=study_sampler(),
                storage=None if no_study_storage else study_storage,
                load_if_exists=not study_reset 
            ) if runner.hyperparameters is not None else None)
            for runner in self.runners
        ]
        
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            self.checkpoint_dir = ".cvrp_benchmarker_checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def _run_trial(
            self,
            client: Client,
            trial: Optional[optuna.Trial],
            runner: Runner,
            check_solutions: bool = False,
            draw_solutions: bool = False,
            draw_cost_histories: bool = False
    ) -> NDArray[numpy.float64]:
        """Runs all the problems with a trial
        :param runner: Runner to run
        :param trial: Trial to use
        :returns: Deviation from optimal
        """
        print(f"Starting run for {runner.name}")
        deviations = numpy.zeros(len(self.problems))
        hyperparameters = None
        if runner.hyperparameters is not None:
            hyperparameters = {
                hyperparameter.name: hyperparameter.get(trial)
                for hyperparameter in runner.hyperparameters
            }
        futures = client.map(
            runner.run, [problem.description for problem in self.problems],
            target_time=self.target_time,
            hyperparameters=hyperparameters
        )
        for problem, result, problem_idx in zip(self.problems, client.gather(futures), range(len(self.problems))):
            if isinstance(result, (float, int)):
                run_result = result
            elif isinstance(result, TrustedRunResult):
                run_result = result.cost
            elif isinstance(result, CheckedRunResult):
                run_result, solution, cost_history = result
                if check_solutions:
                    self.__check_results(runner.name, problem, run_result, solution)
                if draw_solutions:
                    self.__draw_solution(runner.name, run_result, problem, solution)
                if draw_cost_histories:
                    self.__save_cost_history(runner.name, problem, cost_history)
                    
                run_result_idx = cost_history['time'].searchsorted(self.target_time, 'right') - 1
                if run_result_idx == -1:
                    raise TargetTimeExceededException(f'No solution found that completed in time of {self.target_time}. '+
                                                      f'First solution time is {cost_history["time"][0]}')
                run_result = cost_history[run_result_idx]['cost']
            else:
                raise UnknownResultFormat(f"Returned result does not match any of [float, TrustedRunResult, CheckedRunResult]")
                
            deviations[problem_idx] = problem.compare_to_optimal(run_result)
        
        return deviations

    def __check_results(self, runner_name: str, problem: VrplibProblem, dist: float, solution: list[list[int]]) -> bool:
        route_caps = [sum(problem.description.demands[customer] for customer in route) for route in solution]
        exceeded_caps = [cap > problem.description.capacity for cap in route_caps]
        if any(exceeded_caps):
            exceed_indices = [idx for idx, exc in enumerate(exceeded_caps) if exc]
            raise CapacitiesExceededException(f"{runner_name}: Capacities are exceeded for {problem.name} in {exceed_indices}")

        all_customers = set(range(0, problem.description.n_customers))
        customers_in_solution = set([customer for path in solution for customer in path])
        if all_customers != customers_in_solution:
            missing = all_customers - customers_in_solution
            raise NotAllCustomersException(f"{runner_name}: Not all customers found for {problem.name}. Missing: {missing}")

        distances = problem.description.edge_weights
        good_dist = sum(sum(distances[a][b] for a, b in zip(path[1:], path[:-1])) for path in solution)
        if not isclose(dist, good_dist):
            raise WrongCostException(f"{runner_name}: Solution is not close to actual distances. Actual: {good_dist}, found: {dist}")

        return True

    def __save_cost_history(self, runner_name: str, problem: VrplibProblem, history: list[float, float]):
        if history is None:
            return
        history_df = pandas.DataFrame(history)
        history_df.columns = ['time', 'iter', 'cost']
        history_df = history_df.set_index('time')

        history_df.to_csv(os.path.join(self.history_plots_dir, f'{problem.name}.csv'))

        history_df['cost'].plot(title=f'Cost history for "{runner_name}" on problem {problem.name}')
        plt.savefig(os.path.join(self.history_plots_dir, f"{problem.name}.png"))
        plt.close()

    def __draw_solution(self, runner_name: str, dist: float, problem: VrplibProblem, solution: list[list[int]]):
        fig, ax = plt.subplots()
        ax.set_title(f'{problem.name} solved by "{runner_name}" with cost {dist:.3f}')
        numpy.random.seed(0)
        for path in solution:
            path = [problem.node_coord[node] for node in path]
            codes = [Path.MOVETO] + [Path.LINETO for _ in range(1, len(path))]
            path = Path(path, codes)
            patch = patches.PathPatch(path, edgecolor=numpy.random.rand(3, ), facecolor='none', lw=2)
            ax.add_patch(patch)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        plt.savefig(os.path.join(self.solution_plots_dir, f"{problem.name}.png"))
        plt.close(fig)

    def tune_parameters(
            self,
            n_trials: int,
            n_jobs: int = -1,
            concurrency: int = -1,
            cluster = None,
    ) -> None:
        """Tunes hyperparameters for all algorithms with optuna
        :param n_trials: Number of trials for optuna to run
        :param n_jobs: Number of cores/processes to use. Default: -1 (all cores)
        :param concurrency: Number of concurrent calculations. Default: minimum of n_jobs and the number of runners
        :param cluster: A cluster to client a dusk client on. Default: None (Local cluster)
        """
        if concurrency == -1:
            concurrency = len(self.runners)
        if concurrency > len(self.runners):
            print(f"More concurrency ({concurrency}) than the number of runners {len(self.runners)}. Defaulting to the number of runners")
            concurrency = len(self.runners)
        if concurrency > n_jobs:
            print(f"More concurrency ({concurrency}) than the number of jobs {n_jobs}. Defaulting to the number of jobs")
            concurrency = n_jobs

        def launch_study(client: Client, runner: Runner, study: optuna.Study):
            trial = study.ask()
            try:
                trial_results: NDArray[numpy.float64] = self._run_trial(client, trial, runner)
                study.tell(
                    trial=trial,
                    values=trial_results.mean(),
                    state=optuna.trial.TrialState.COMPLETE
                )
            except optuna.exceptions.TrialPruned as e:
                print(e)
                study.tell(trial=trial, state=optuna.trial.TrialState.PRUNED)
            except Exception:
                study.tell(trial=trial, state=optuna.trial.TrialState.FAIL)
        with (
            LocalCluster(
                'tcp://localhost:9895',
                n_workers=n_jobs,
                processes=True,
                threads_per_worker=1
            ) if cluster is None else nullcontext(cluster) as cluster,
            Client(cluster, asynchronous=True) as client,
            ThreadPoolExecutor(concurrency) as executor
        ):
            futures = []
            for _ in range(n_trials):
                for runner, study in self.studies:
                    if study is not None:
                        futures.append(executor.submit(launch_study, client, runner, study))
            
            for future in futures:
                future.result()
     
    def benchmark(
            self,
            n_runs: int,
            n_jobs: int = -1,
            concurrency: int = -1,
            cluster = None,
            load_checkpoint: bool = True,
    ) -> pandas.DataFrame:
        """Benchmarks the problems with *n_runs* consecutive runs.

        :param n_runs: Number of runs for benchmarking
        :param n_jobs: Number of cores/processes to use. Default: -1 (all cores)
        :param concurrency: Number of concurrent calculations. Default: minimum of n_jobs and the number of runners
        :param cluster: A cluster to client a dusk client on. Default: None (Local cluster)
        :param load_checkpoint: Attempt to load the checkpoint file or not. Default: True
        """
        checkpoint_filename = f"benchmark_{'_'.join(runner.name for runner in self.runners)}_{n_runs}.csv"
        checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_filename)

        csv_output_lock = threading.Lock()
        if load_checkpoint and os.path.exists(checkpoint_file):
            deviations_df = pandas.read_csv(checkpoint_file, index_col=[0, 1], skipinitialspace=True)
        else:
            deviations_df = pandas.DataFrame(
                index=pandas.MultiIndex.from_product(
                    [[problem.name for problem in self.problems], range(1, n_runs + 1)],
                    names=['problem', 'iteration']
                ),
                columns=[runner.name for runner in self.runners]
            )

        if concurrency == -1:
            concurrency = len(self.runners)
        if concurrency > len(self.runners):
            print(f"More concurrency ({concurrency}) than the number of runners {len(self.runners)}. Defaulting to the number of runners")
            concurrency = len(self.runners)
        if concurrency > n_jobs:
            print(f"More concurrency ({concurrency}) than the number of jobs {n_jobs}. Defaulting to the number of jobs")
            concurrency = n_jobs

        def launch_run(client: Client, runner: Runner, study: Optional[optuna.Study], run_num: int):
            trial = None
            if study is not None:
                if len(study.best_trials) == 0:
                    trial = study.ask()
                else:
                    trial = study.best_trial
            
            trial_results: NDArray[numpy.float64] = self._run_trial(client, trial, runner)
            with csv_output_lock:
                deviations_df[runner.name][:][run_num-1] = trial_results
                deviations_df.to_csv(checkpoint_file)
        with (
            LocalCluster(
                'tcp://localhost:9895',
                n_workers=n_jobs,
                processes=True,
                threads_per_worker=1
            ) if cluster is None else nullcontext(cluster) as cluster,
            Client(cluster, asynchronous=True) as client,
            ThreadPoolExecutor(concurrency) as executor
        ):
            futures = []
            for i in range(1, n_runs + 1):
                for runner, study in self.studies:
                    if (not load_checkpoint) or (load_checkpoint and pandas.isna(deviations_df[runner.name][self.problems[0].name][i])):
                        futures.append(executor.submit(launch_run, client, runner, study, i))
            for future in futures:
                future.result()

        return deviations_df
        