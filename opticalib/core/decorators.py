"""
Decorators used across opticalib.

This module collects reusable decorators that add cross-cutting behavior to
functions without changing their core implementation.
"""

from __future__ import annotations

from functools import wraps
from inspect import Signature, signature
from typing import Any, Callable, TypeVar
from .exceptions import ReconnectionError

try:
    from typing import ParamSpec
except ImportError:  # pragma: no cover
    from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


def expand_list_arguments(
    param_names: list[str],
    *,
    strict_length: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R | list[R]]]:
    """
    Expand list-valued parameters into multiple single-value function calls.

    This decorator reproduces the common pattern where a function accepts either
    a scalar argument or a list of arguments. If at least one of ``param_names``
    is a list, then all tracked parameters must be lists; mixed scalar/list
    tracked inputs are rejected. In list mode, the wrapped function is called
    once per list element, replacing list arguments with their ``i``-th element.

    Parameters
    ----------
    param_names : list[str]
            Names of parameters to inspect for list-expansion behavior.
    strict_length : bool, optional
            If True, all list arguments among ``param_names`` must have the same
            length; otherwise, a ``ValueError`` is raised. The default is True.

    Returns
    -------
    Callable
            A decorator that wraps the target function.

    Raises
    ------
    KeyError
            If one or more ``param_names`` are not present in the target function
            signature.
    ValueError
            If tracked parameters mix list and non-list values in the same call.
    ValueError
            If ``strict_length`` is True and detected list arguments have different
            lengths.

    Examples
    --------
    >>> @expand_list_arguments(["tn"])
    ... def process(tn: str, save: bool = False) -> str:
    ...     return f"Processed {tn}"
    >>> process("20260101_000000")
    'Processed 20260101_000000'
    >>> process(["20260101_000000", "20260101_000001"])
    ['Processed 20260101_000000', 'Processed 20260101_000001']
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R | list[R]]:
        sig = signature(func)

        missing = [name for name in param_names if name not in sig.parameters]
        if missing:
            msg = "Parameter(s) not found in function signature: " + ", ".join(missing)
            raise KeyError(msg)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | list[R]:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            is_list_flags: list[bool] = []
            for name in param_names:
                value = bound.arguments.get(name)
                is_list_flags.append(isinstance(value, list))

            if any(is_list_flags) and not all(is_list_flags):
                raise ValueError(
                    "Tracked parameters must be all lists or all scalars. "
                    "Mixed list/scalar values are not allowed."
                )

            list_lengths: list[int] = []
            for name in param_names:
                value = bound.arguments.get(name)
                if isinstance(value, list):
                    list_lengths.append(len(value))

            if not list_lengths:
                return func(*args, **kwargs)

            if strict_length and len(set(list_lengths)) != 1:
                raise ValueError(
                    "All list-valued parameters must have the same length."
                )

            n_calls = list_lengths[0]
            results: list[R] = []

            for i in range(n_calls):
                call_arguments = dict(bound.arguments)
                for name in param_names:
                    value = call_arguments.get(name)
                    if isinstance(value, list):
                        call_arguments[name] = value[i]

                call_args, call_kwargs = _build_call_arguments(sig, call_arguments)
                results.append(func(*call_args, **call_kwargs))

            return results

        return wrapper

    return decorator


def allow_reconnect(
    max_retries: int = 5,
    error_instance: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that automatically attempts to reconnect to a hardware which has
    a `reconnect` method when a specified exception is raised during the execution
    of the decorated function.

    Parameters
    ----------
    max_retries : int, optional
            Maximum number of reconnection attempts before raising
            ReconnectionError. The default is 5.
    error_instances : tuple of Exception type, optional
            The specific exception type that triggers a reconnection attempt.
            The default is the base Exception class, which will catch all exceptions.

    Returns
    -------
    Callable
            A decorator that wraps the target method.

    Raises
    ------
    ReconnectionError
            If the camera cannot be reconnected after all retry attempts.

    Examples
    --------
    >>> class VmbCamera:
    ...     @vmbpy_reconnect(max_retries=3)
    ...     def get_frame(self):
    ...         # Implementation that might raise VmbFeatureError
    ...         return self.cam.get_frame()
    """
    import time

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:

            attempt = 0
            last_error = None

            while attempt <= max_retries:
                try:
                    return func(*args, **kwargs)
                except error_instance as e:
                    last_error = e
                    if attempt < max_retries:
                        attempt += 1
                        # Call reconnect on self (assumes first arg is self)
                        if args:
                            self_obj = args[0]
                            if hasattr(self_obj, "reconnect"):
                                try:
                                    self_obj.reconnect()
                                    time.sleep(0.1 * attempt)  # Backoff delay
                                except Exception:
                                    pass
                    else:
                        break

            raise ReconnectionError(
                f"Failed to reconnect after {max_retries} attempts. "
                f"Original error: {last_error}"
            ) from last_error

        return wrapper

    return decorator


def _build_call_arguments(
    sig: Signature,
    arguments: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """
    Rebuild positional and keyword arguments from a bound argument mapping.

    Parameters
    ----------
    sig : Signature
            Signature of the wrapped function.
    arguments : dict[str, Any]
            Mapping produced by ``Signature.bind_partial``.

    Returns
    -------
    args : list[Any]
            Positional arguments for the function call.
    kwargs : dict[str, Any]
            Keyword arguments for the function call.
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name not in arguments:
            continue

        value = arguments[name]
        if param.kind in (
            param.POSITIONAL_ONLY,
            param.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        elif param.kind is param.VAR_POSITIONAL:
            args.extend(value)
        elif param.kind is param.KEYWORD_ONLY:
            kwargs[name] = value
        elif param.kind is param.VAR_KEYWORD:
            kwargs.update(value)

    return args, kwargs


__all__ = ["expand_list_arguments", "vmbpy_reconnect", "ReconnectionError"]
