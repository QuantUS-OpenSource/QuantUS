from pathlib import Path

from argparse import ArgumentParser

from .functions import *

def get_im_preproc_funcs() -> dict:
    """Get preprocessing functions for the CLI.

    Returns:
        dict: Dictionary of preprocessing functions.
    """
    functions = {name: obj for name, obj in globals().items() if callable(obj) and obj.__module__ == 'src.image_preprocessing.functions'}
    return functions

def get_required_im_preproc_kwargs(preproc_func_names: list) -> list:
    """Get required kwargs for a given list of preprocessing functions.

    Args:
        preproc_func_names (list): list of preprocessing function names to apply.

    Returns:
        list: List of required kwargs for the specified preprocessing functions.
    """
    preproc_funcs = get_im_preproc_funcs()
    required_kwargs = []

    for func_name in preproc_func_names:
        func = preproc_funcs[func_name]
        required_kwargs.extend(getattr(func, 'required_kwargs', []))
    
    required_kwargs = list(set(required_kwargs))  # Remove duplicates
    return required_kwargs
