import os
from tempfile import TemporaryDirectory

from dataclasses import dataclass

from numpy.typing import NDArray
import numpy

import vrplib


@dataclass(frozen=True)
class CvrpProblemDescription:
    """Represents data passed to a solver.
    :ivar n_customers: Number of customers
    :type n_customers: int

    :ivar capacity: Capacity of a single vehicle
    :type capacity: int

    :ivar demands: Demands of customers in an ndarray
    :type demands: numpy.ndarray(shape=(n_customers,), dtype=int)

    :ivar coords: Coordinates of all points. First one is depot, the rest are customers
    :type coords: numpy.ndarray(shape=(n_customers, 2), dtype=float)

    :ivar edge_weights: Matrix of distances between points
    :type edge_weights: numpy.ndarray(shape=(n_customers, n_customers), dtype=float)
    """
    n_customers: int

    capacity: int
    demands: NDArray[numpy.int64]

    coords: NDArray[numpy.float64]
    edge_weights: NDArray[numpy.float64]


class VrplibProblem:
    """Loads problems and solutions from vrplib.
    Produces problem description and compares run results to an optimal solution
    """

    def __init__(
            self,
            problem_name: str,
            problems_dir: str = None
    ):
        """
        :param problem_name: The name of a problem to load
        :param problems_dir: Path to directory for storing problems.
            Default: None (Problems will be downloaded to a temporary directory)
        """
        if problems_dir is None:
            with TemporaryDirectory() as prob_dir:
                instance_path = os.path.join(prob_dir, f'{problem_name}.ins')
                solution_path = os.path.join(prob_dir, f'{problem_name}.sol')
                vrplib.download_instance(problem_name, instance_path)
                vrplib.download_solution(problem_name, solution_path)
                ins = vrplib.read_instance(instance_path)
                sol = vrplib.read_solution(solution_path)
        else:
            if not os.path.exists(problems_dir):
                os.mkdir(problems_dir)
            instance_path = os.path.join(problems_dir, f'{problem_name}.ins')
            solution_path = os.path.join(problems_dir, f'{problem_name}.sol')
            if not os.path.exists(instance_path):
                vrplib.download_instance(problem_name, instance_path)
            if not os.path.exists(solution_path):
                vrplib.download_solution(problem_name, solution_path)
            ins = vrplib.read_instance(instance_path)
            sol = vrplib.read_solution(solution_path)
            
        self.name = ins['name']
        self.comment = ins['comment']

        self.description = CvrpProblemDescription(
            n_customers=ins['dimension'],
            capacity=ins['capacity'],
            demands=ins['demand'],
            coords=ins['node_coord'],
            edge_weights=ins['edge_weight']
        )

        self.best_known_cost = sol['cost']
        self.best_known_routes = sol['routes']

    def compare_to_optimal(self, cost: float) -> float:
        """
        :param cost: Cost of solution you would like to compare to optimal
        :return: Submitted cost divided by best known cost
        """
        return cost / self.best_known_cost
    