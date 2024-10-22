"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, List, Tuple, Optional  # Added Optional import

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            # print(v)
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the negative of the input tensor."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the negate operation."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the inverse of the input tensor."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the inverse operation."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise addition of two tensors."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the addition operation.

        Args:
        ----
            ctx: The context object containing saved values.
            grad_output: The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the input tensors.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors."""
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)  # Use the backend's multiplication method

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for element-wise multiplication."""
        (a, b) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, b), grad_output.f.mul_zip(
            grad_output, a
        )  # Use backend for gradients


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Applies the sigmoid function element-wise."""
        ctx.save_for_backward(a)
        return a.f.sigmoid_map(a)  # Use the backend's sigmoid method

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the sigmoid function."""
        a = ctx.saved_tensors[0]
        sigmoid_a = a.f.sigmoid_map(a)  # Forward sigmoid value

        # Ensure 1.0 is a tensor for backend compatibility
        one_tensor = a._ensure_tensor(1.0)

        # Compute sigmoid derivative: sigmoid(a) * (1 - sigmoid(a))
        sigmoid_derivative = sigmoid_a.f.mul_zip(
            sigmoid_a, sigmoid_a.f.add_zip(one_tensor, a.f.neg_map(sigmoid_a))
        )

        # Return grad_output * sigmoid derivative
        return grad_output.f.mul_zip(grad_output, sigmoid_derivative)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Applies the ReLU function element-wise."""
        ctx.save_for_backward(a)
        return a.f.relu_map(a)  # Use the backend's ReLU method

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the ReLU function."""
        a = ctx.saved_values[0]
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes the natural logarithm of the input tensor."""
        ctx.save_for_backward(a)
        return a.f.log_map(a)  # Use the backend's log method

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the logarithm function."""
        a = ctx.saved_values[0]
        return grad_output.f.log_back_zip(a, grad_output)  # Use backend for gradients


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes the exponential of the input tensor."""
        ctx.save_for_backward(a)
        return a.f.exp_map(a)  # Use the backend's exp method

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the exponential function."""
        a = ctx.saved_values[0]
        return grad_output.f.mul_zip(
            grad_output, a.f.exp_map(a)
        )  # Use backend for gradients


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the sum of the tensor elements along the specified dimension."""
        ctx.save_for_backward(a, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for summation."""
        # a = ctx.saved_tensors[0]  # Get the saved tensor from forward
        return (grad_output, 0.0)  # Use backend for gradients


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise less than comparison."""
        return a.f.lt_zip(a, b)  # Use the backend's less than method

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[float, float]:
        """Backward pass for less than comparison (no gradient)."""
        return (0.0, 0.0)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Element-wise equality comparison."""
        return a.f.eq_zip(a, b)  # Use the backend's equality method

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[float, float]:
        """Backward pass for equality comparison (no gradient)."""
        return (0.0, 0.0)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Checks if two tensors are element-wise equal within a tolerance."""
        return a.f.is_close_zip(a, b)  # Use the backend's is close method


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dims: Tensor) -> Tensor:
        """Permutes the dimensions of the input tensor according to the specified order."""
        dims_tuple = tuple(int(d) for d in dims._tensor._storage)
        ctx.save_for_backward(dims_tuple)
        return a._new(a._tensor.permute(*dims_tuple))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for permuting dimensions."""
        dims_tuple = ctx.saved_values[0]

        reverse_dims = [0] * len(dims_tuple)
        for i, dim in enumerate(dims_tuple):
            reverse_dims[dim] = i

        # un-permute
        grad_input = grad_output.permute(*reverse_dims)

        return (grad_input, 0.0)


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshape the input tensor to the specified shape.

        Args:
        ----
            ctx: The context object for saving information.
            a: The input tensor to reshape.
            shape: The desired shape for the output tensor.

        Returns:
        -------
            Tensor: A new tensor with the specified shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the gradient of a function using central difference approximation.

    Args:
    ----
        f: The function for which the gradient is computed.
        vals: The input tensors to the function.
        arg: The index of the argument to compute the gradient for.
        epsilon: The small value used for the central difference.
        ind: The index in the tensor to perturb.

    Returns:
    -------
        float: The approximated gradient value.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
