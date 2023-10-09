from abc import ABC, abstractmethod
from dataclasses import dataclass
import optuna
import numpy

from typing import NamedTuple, Union, Sequence, Optional, Mapping
from numpy.typing import NDArray

from cvrp_benchmarker.problem import CvrpProblemDescription


@dataclass(frozen=True)
class HyperParameter:
    """Hyperparameter used for optuna
    :ivar name: Name of a hyperparameter in optuna
    :type name: str

    :ivar bounds: Bounds for the hyperparameter
    :type bounds: Union[tuple[float, float], tuple[int, int]]
    """
    name: str
    bounds: tuple
    log: bool = False

    def __repr__(self):
        return self.name

    def get(
            self,
            trial: optuna.Trial
    ) -> Union[int, float]:
        if isinstance(self.bounds[0], float):
            return trial.suggest_float(
                self.name,
                self.bounds[0], self.bounds[1],
                log=self.log
            )
        else:
            return trial.suggest_int(
                self.name,
                self.bounds[0], self.bounds[1],
                log=self.log
            )


class TrustedRunResult(NamedTuple):
    """Result of an algorithms's run on a problem, accepted with no checks

    :ivar cost: Total route length of a solution
    :type cost: float
    """
    cost: float


cost_history_entry = numpy.dtype([('time', numpy.float64), ('iteration', numpy.int64), ('cost', numpy.float64)])

class CheckedRunResult(TrustedRunResult):
    """Result of an algorithms's run on a problem, checked for validity

    :ivar cost: Total route length of a solution
    :type cost: float
    
    :ivar routes: Routes created by a solution
    :type routes: Sequence[Sequence[int]]
    
    :ivar cost_history: History of cost changes
    :type cost_history: numpy.dtype([('time', 'f64'), ('iteration', 'i64'), ('cost', 'f64')])
    """
    routes: Sequence[Sequence[int]]
    cost_history: NDArray[cost_history_entry]


class Runner(ABC):
    """Wrapper-class for an algorithm and its hyperparameters.
    Specify hyperparemeters on init and use them in run.
    Omit hyperparameters, if your algorithm has none.
    """

    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        """The display name of the class (defaults to class name)"""
        return cls.__name__

    @classmethod
    @property
    @abstractmethod
    def hyperparameters(cls) -> Optional[Sequence[HyperParameter]]:
        """hyperparameters that need tuning"""
        pass

    @classmethod
    @abstractmethod
    def run(
            cls,
            problem: CvrpProblemDescription,
            target_time: float,
            hyperparameters: Optional[Mapping[str, Union[float, int]]]
    ) -> Union[float, TrustedRunResult, CheckedRunResult]:
        """Run an algorithm on a certain problem with specified hyperparameters and a limit of time.

        :param problem: Problem description to solve.
        :param target_time: Time expected for the algorythm to finish.
        :param hyperparameters: hyperparameters used for this run as a mapping from hyperparamete name to
                                suggested value
        
        :return: one of [float, TrustedRunResult, CheckedRunResult]
            - float: treated the same as TrustedRunResult
            - TrustedRunResult: a named tuple of (float), the results gets accepted blindly.
            - CheckedRunResult: gets checked before getting accepted, a named tuple of (float, Sequence[Sequence[int]], Sequence[Tuple[Union[float, int], int, float]]): 
                - Minimal length of routes found;
                - Routes themselves as sequences of nodes represented by node index in the problem description
                - Cost history as a sequence of (time in seconds, iteration, cost) entries
                Gets checked before getting accepted
        """
        pass
