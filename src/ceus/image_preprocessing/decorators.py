from typing import List

def required_kwargs(*kwarg_names: List[str]):
    """
    A decorator to specify the required keyword arguments for a function.

    Args:
        kwarg_names (list): List of required keyword argument names.

    Returns:
        function: The decorated function with the specified keyword arguments attached as metadata.
    """
    def decorator(func):
        func.required_kwargs = kwarg_names
        return func
    return decorator
