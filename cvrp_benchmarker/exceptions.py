from optuna.exceptions import TrialPruned


class NotAllCustomersException(TrialPruned):
    pass


class CapacitiesExceededException(TrialPruned):
    pass


class WrongCostException(TrialPruned):
    pass


class TargetTimeExceededException(TrialPruned):
    pass


class UnknownResultFormat(TrialPruned):
    pass
