import os
from math import isclose

import numpy
import pandas

from matplotlib import pyplot as plt, patches
from matplotlib.path import Path

from cvrp_benchmarker.problem import VrplibProblem
from cvrp_benchmarker.exceptions import (
    CapacitiesExceededException,
    NotAllCustomersException,
    WrongCostException
) 


def check_results(
    runner_name: str,
    problem: VrplibProblem,
    dist: float,
    solution: list[list[int]]
) -> bool:
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


def save_cost_history(
    runner_name: str,
    problem: VrplibProblem,
    history: list[float, float],
    plot_dir: str
):
    if history is None:
        return
    history_df = pandas.DataFrame(history)
    history_df.columns = ['time', 'iter', 'cost']
    history_df = history_df.set_index('time')

    history_df.to_csv(os.path.join(plot_dir, f'{problem.name}.csv'))

    history_df['cost'].plot(title=f'Cost history for "{runner_name}" on problem {problem.name}')
    plt.savefig(os.path.join(plot_dir, f"{problem.name}.png"))
    plt.close()


def draw_solution(
    runner_name: str,
    dist: float,
    problem: VrplibProblem,
    solution: list[list[int]],
    plot_dir: str
):
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
    plt.savefig(os.path.join(plot_dir, f"{problem.name}.png"))
    plt.close(fig)
