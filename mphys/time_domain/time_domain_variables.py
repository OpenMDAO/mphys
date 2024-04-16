from dataclasses import dataclass

@dataclass
class TimeDerivativeVariable:
    """
    Description of a variable that needs backplanes of data
    """
    name: str
    number_of_backplanes: int
    shape: tuple
    distributed: bool = False


@dataclass
class TimeDomainInput:
    """
    Description of a variable that needs to be an input to subsystems in a time step
    """
    name: str
    shape: tuple
    distributed: bool = False
