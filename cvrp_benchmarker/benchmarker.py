import os
from contextlib import nullcontext

from typing import Optional, Sequence, Type

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import optuna
from dask.distributed import Client

import numpy
from numpy.typing import NDArray
import pandas
import vrplib

from cvrp_benchmarker.problem import VrplibProblem
from cvrp_benchmarker.runner import Runner, CheckedRunResult, TrustedRunResult
from cvrp_benchmarker.exceptions import TargetTimeExceededException, UnknownResultFormat
from cvrp_benchmarker._utility import check_results, save_cost_history, draw_solution
from cvrp_benchmarker.cluster_pool import ClusterPool


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
        print(f"Starting run for {runner.name}")
        deviations = numpy.zeros(len(self.problems))
        hyperparameters = None
        if runner.hyperparameters is not None:
            hyperparameters = {
                hyperparameter.name: hyperparameter.get(trial)
                for hyperparameter in runner.hyperparameters
            }
        
        futures = client.map(
            partial(runner.run, target_time=self.target_time, hyperparameters=hyperparameters),
            [problem.description for problem in self.problems],
        )
        for problem, result, problem_idx in zip(self.problems, client.gather(futures), range(len(self.problems))):
            if isinstance(result, (float, int)):
                run_result = result
            elif isinstance(result, TrustedRunResult):
                run_result = result.cost
            elif isinstance(result, CheckedRunResult):
                run_result, solution, cost_history = result
                if check_solutions:
                    check_results(runner.name, problem, run_result, solution)
                if draw_solutions:
                    draw_solution(runner.name, run_result, problem, solution, self.history_plots_dir)
                if draw_cost_histories:
                    save_cost_history(runner.name, problem, cost_history, self.solution_plots_dir)
                    
                run_result_idx = cost_history['time'].searchsorted(self.target_time, 'right') - 1
                if run_result_idx == -1:
                    raise TargetTimeExceededException(f'No solution found that completed in time of {self.target_time}. '+
                                                      f'First solution time is {cost_history["time"][0]}')
                run_result = cost_history[run_result_idx]['cost']
            else:
                raise UnknownResultFormat(f"Returned result does not match any of [float, TrustedRunResult, CheckedRunResult]")
                
            deviations[problem_idx] = problem.compare_to_optimal(run_result)
        
        return deviations

    def tune_parameters(
            self,
            n_trials: int,
            cluster_pool: Optional[ClusterPool],
    ) -> None:
        """Tunes hyperparameters for all algorithms with optuna
        :param n_trials: Number of trials for optuna to run
        :param n_jobs: Number of cores/processes to use. Default: -1 (all cores)
        :param concurrency: Number of concurrent calculations. Default: minimum of n_jobs and the number of runners
        :param cluster: A cluster to client a dusk client on. Default: None (Local cluster)
        """

        def launch_study(client: Client, runner: Runner, study: optuna.Study):
            trial = study.ask()
            try:
                trial_results: NDArray[numpy.float64] = self._run_trial(client, trial, runner, check_solutions=True)
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
            ClusterPool() if cluster_pool is None else nullcontext(cluster_pool) as cluster_pool,
            ThreadPoolExecutor(cluster_pool.size()) as executor,
        ):
            futures = []
            for i in range(n_trials):
                for runner, study in self.studies:
                    if study is not None:
                        futures.append(executor.submit(
                            launch_study,
                            cluster_pool.next(),
                            runner,
                            study
                        ))
            
            for future in futures:
                future.result()
     
    def benchmark(
            self,
            n_runs: int,
            cluster_pool: Optional[ClusterPool],
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
            ClusterPool() if cluster_pool is None else nullcontext(cluster_pool) as cluster_pool,
            ThreadPoolExecutor(cluster_pool.size()) as executor,
        ):
            futures = []
            for i in range(1, n_runs + 1):
                for runner, study in self.studies:
                    if (not load_checkpoint) or (
                            load_checkpoint and
                            pandas.isna(deviations_df[runner.name][self.problems[0].name][i])
                        ):
                        futures.append(executor.submit(launch_run, cluster_pool.next(), runner, study, i))
            for future in futures:
                future.result()

        return deviations_df
        