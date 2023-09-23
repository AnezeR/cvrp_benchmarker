from typing import Optional, Sequence
from cvrp_benchmarker import Runner, HyperParameter


def test_Runner_init_name():
    class MockRunner(Runner):
        @classmethod
        @property
        def name(cls) -> str:
            return 'some_name'
        @classmethod
        def run(cls):
            pass
    runner = MockRunner
    assert runner.name == 'some_name'

    class MockRunner(Runner):
        @classmethod
        def run(cls):
            pass
    runner = MockRunner
    assert runner.name == 'MockRunner'

def test_Runner_init_params():
    class MockRunner(Runner):
        @classmethod
        @property
        def hyperparameters(cls) -> Optional[Sequence[HyperParameter]]:
            return [
                HyperParameter('a', (1, 2)),
                HyperParameter('b', (1., 2.))
            ]
        @classmethod
        def run(cls):
            pass

    runner = MockRunner
    assert runner.hyperparameters[0].bounds[0].__class__ != float
    assert runner.hyperparameters[1].bounds[1].__class__ == float
