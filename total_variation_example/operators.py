import functools
import torch
from abc import ABC, abstractmethod
from torch import nn
from typing import Any, Tuple, Dict, Type

class BaseLinearOperator(nn.Module, ABC):
    """
    Base class for linear operators in a neural network module.
    This abstract base class defines the interface for linear operators, including methods
    for obtaining the transpose of the operator, performing forward operations, and matrix
    multiplication using the @ operator. Subclasses must implement the `get_transpose_class`
    and `forward` methods.
    Attributes:
        _args (Tuple[Any, ...]): Variable length argument list.
        _kwargs (Dict[str, Any]): Arbitrary keyword arguments.
    Methods:
        get_transpose_class() -> Type['BaseLinearOperator']:
        forward(argument: torch.Tensor) -> torch.Tensor:
        __matmul__(argument: torch.Tensor) -> torch.Tensor:
        __repr__() -> str:
        invalidate_cache() -> None:
        T:
    """

    @classmethod
    @abstractmethod
    def get_transpose_class(cls) -> Type['BaseLinearOperator']:
        """
        Returns the class of the transposed operator. Must be implemented by subclasses.

        Returns:
            Type[BaseLinearOperator]: The transposed operator class.
        """
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialise the instance with given arguments and keyword arguments.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
        """
        super().__init__()
        self._args: Tuple[Any, ...] = args
        self._kwargs: Dict[str, Any] = kwargs

    @abstractmethod
    def forward(self, argument: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward operation to the given argument. Must be implemented by subclasses.

        Args:
            argument (torch.Tensor): The input tensor on which the forward operation is performed.

        Returns:
            torch.Tensor: The result of the forward operation.
        """
        pass

    def __matmul__(self, argument: torch.Tensor) -> torch.Tensor:
        """
        Perform matrix multiplication using the @ operator.

        This method allows the use of the @ operator to perform matrix multiplication
        by calling the forward method of the class.

        Args:
            argument (torch.Tensor): The tensor to be multiplied.

        Returns:
            torch.Tensor: The result of the matrix multiplication.
        """
        return self.forward(argument)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        This method returns a string that includes the class name and the values
        of the '_args' and '_kwargs' attributes. It is useful for debugging and
        logging purposes.

        Returns:
            str: A string representation of the object.
        """
        return f"{self.__class__.__name__}(args={self._args}, kwargs={self._kwargs})"

    def invalidate_cache(self) -> None:
        """
        Invalidates the cached transpose operator if it exists.

        This method checks if the transpose operator 'T' is present in the instance's
        dictionary. If it is, the method deletes it from the dictionary, effectively
        invalidating the cached value.

        Returns:
            None
        """
        if 'T' in self.__dict__:
            del self.__dict__['T']

    @functools.cached_property
    def T(self) -> 'BaseLinearOperator':
        """
        Returns the transpose of the current operator.

        This method retrieves the class representing the transpose of the current operator
        by calling `get_transpose_class` and then instantiates it with the same arguments
        and keyword arguments as the current operator.

        Returns:
            An instance of the transpose class of the current operator.
        """
        transpose_cls = self.get_transpose_class()
        return transpose_cls(*self._args, **self._kwargs)

class Forward_Finite_Differences_1D(BaseLinearOperator):
    """
    A class to represent a finite difference operator in 1D.
    Attributes
    ----------
    step_size : float, optional
        The step size for the finite difference calculation (default is 1).
    transpose : bool, optional
        A flag to indicate whether to apply the transpose of the operator (default is False).
    Methods
    -------
    __call__(argument)
        Applies the finite difference operator to the given argument.
        If transpose is True, applies the transpose of the operator.
    """

    @classmethod
    def get_transpose_class(cls) -> Type['BaseLinearOperator']:
        """
        Returns the class of the one-dimensional backward finite differences.

        Returns:
            Type[BaseLinearOperator]: The Backward_Finite_Differences_1D class.
        """
        return Backward_Finite_Differences_1D

    def __init__(self, step_size=1):        
        super().__init__(step_size)

    def forward(self, argument: torch.Tensor) -> torch.Tensor:
        return (argument[:, :, 1:] - argument[:, :, :-1]) / self._args[0]
        
class Backward_Finite_Differences_1D(BaseLinearOperator):

    @classmethod
    def get_transpose_class(cls) -> Type['BaseLinearOperator']:
        """
        Returns the class of the one-dimensional forward finite differences.

        Returns:
            Type[BaseLinearOperator]: The Forward_Finite_Differences_1D class.
        """
        return Forward_Finite_Differences_1D
    
    def __init__(self, step_size=1):        
        super().__init__(step_size)

    def forward(self, argument: torch.Tensor) -> torch.Tensor:
        first_element = -argument[:, :, 0].unsqueeze(-1)
        middle_elements = (argument[:, :, :-1] - argument[:, :, 1:])
        last_element = argument[:, :, -1].unsqueeze(-1)
        output = torch.cat((first_element, middle_elements, last_element), dim=-1)
        return output / self._args[0]
    
class Finite_Difference_Gradient_2D(BaseLinearOperator):

    @classmethod
    def get_transpose_class(cls) -> Type['BaseLinearOperator']:
        """
        Returns the class of the two-dimensional finite-difference divergence.

        Returns:
            Type[BaseLinearOperator]: The Finite_Difference_Divergence_2D class.
        """
        return Finite_Difference_Divergence_2D
    
    def __init__(self, step_sizes=(1, 1)) -> None:
        super().__init__(step_sizes)

    def forward(self, argument: torch.Tensor) -> torch.Tensor:
        no_batches, no_channels, no_rows, no_columns = argument.shape
        output = torch.zeros(no_batches, 2*no_channels, no_rows, no_columns, dtype=argument.dtype, device=argument.device)
        output[:, 0:no_channels, :, :-1] = (argument[:, 0:no_channels, :, 1:] - argument[:, 0:no_channels, :, :-1]) / self._args[0][0]
        output[:, no_channels:(2*no_channels), :-1, :] = (argument[:, 0:no_channels, 1:, :] - argument[:, 0:no_channels, :-1, :]) / self._args[0][1]
        return output

class Finite_Difference_Divergence_2D(BaseLinearOperator):

    @classmethod
    def get_transpose_class(cls) -> Type['BaseLinearOperator']:
        """
        Returns the class of the two-dimensional finite-difference gradient.

        Returns:
            Type[BaseLinearOperator]: The Finite_Difference_Gradient_2D class.
        """
        return Finite_Difference_Gradient_2D
    
    def __init__(self, step_sizes=(1, 1)) -> None:
        super().__init__(step_sizes)

    def forward(self, argument: torch.Tensor) -> torch.Tensor:
        no_batches, no_channels, no_rows, no_columns = argument.shape
        output = torch.zeros(no_batches, no_channels // 2, no_rows, no_columns, dtype=argument.dtype, device=argument.device)
        
        # Handle the x-direction (first half of channels)
        first_element_x = -argument[:, 0:no_channels//2, :, 0].unsqueeze(-1)
        middle_elements_x = (argument[:, 0:no_channels//2, :, :-2] - argument[:, 0:no_channels//2, :, 1:-1])
        last_element_x = argument[:, 0:no_channels//2, :, -2].unsqueeze(-1)
        output += torch.cat((first_element_x, middle_elements_x, last_element_x), dim=-1) / self._args[0][0]
        
        # Handle the y-direction (second half of channels)
        first_element_y = -argument[:, no_channels//2:, 0, :].unsqueeze(-2)
        middle_elements_y = (argument[:, no_channels//2:, :-2, :] - argument[:, no_channels//2:, 1:-1, :])
        last_element_y = argument[:, no_channels//2:, -2, :].unsqueeze(-2)
        output += torch.cat((first_element_y, middle_elements_y, last_element_y), dim=-2) / self._args[0][1]
        return output
    
class Identity_Operator(BaseLinearOperator):
    """
    A class representing the identity operator, which returns the input argument unchanged.

    Inherits from:
        BaseLinearOperator

    Attributes:
        transpose (bool): A flag indicating whether to use the transpose of the operator. Default is False.

    Methods:
        __call__(argument):
            Applies the identity operator to the input argument and returns it unchanged.
            Args:
                argument: The input to which the identity operator is applied.
            Returns:
                The input argument unchanged.
    """

    @classmethod
    def get_transpose_class(cls) -> Type['BaseLinearOperator']:
        """
        Returns the identity operator class.

        Returns:
            Type[BaseLinearOperator]: The Identity_Operator class.
        """
        return Identity_Operator

    def __init__(self):
        super().__init__()

    def forward(self, argument: torch.Tensor) -> torch.Tensor:
        return argument