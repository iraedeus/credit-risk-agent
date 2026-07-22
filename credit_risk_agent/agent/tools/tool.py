import inspect
from collections.abc import Callable
from typing import Any

from gigachat.models import Function, FunctionParameters


class Tool:
    """
    Wrapper class for agent tools with GigaChat function conversion support.

    Encapsulates a callable Python function along with its metadata (name and
    description) extracted automatically from its docstring if not provided.

    Parameters
    ----------
    func : Callable[..., Any]
        The Python function or callable to wrap as an agent tool.
    name : str or None, default=None
        The unique name of the tool. If None, `func.__name__` is used.
    description : str or None, default=None
        A description of the tool's purpose. If None, extracts the docstring
        of `func`.

    Attributes
    ----------
    func : Callable[..., Any]
        The wrapped callable object.
    name : str
        The tool name used in LLM function calling schemas.
    description : str
        The description of the tool provided to the LLM.
    """

    def __init__(self, func: Callable[..., Any], name: str | None = None, description: str | None = None) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""

    def to_gigachat_function(self, parameter_schema: dict[str, Any]) -> Function:
        """
        Convert the tool into a GigaChat `Function` model instance.

        Parameters
        ----------
        parameter_schema : dict[str, Any]
            The JSON Schema dictionary defining the expected function arguments.

        Returns
        -------
        Function
            A GigaChat `Function` object configured for function calling API requests.
        """
        return Function(name=self.name, description=self.description, parameters=FunctionParameters(**parameter_schema))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the wrapped function with the given arguments.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the wrapped function.
        **kwargs : Any
            Keyword arguments passed to the wrapped function.

        Returns
        -------
        Any
            The result returned by the wrapped function.
        """
        return self.func(*args, **kwargs)
