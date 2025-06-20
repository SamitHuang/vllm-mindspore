import inspect
from collections import OrderedDict

import mindspore
from mindspore import mutable, nn


def get_tensor_dynamic_input(tensors):
    if tensors is None:
        return None
    elif isinstance(tensors, mindspore.Tensor):
        return mindspore.Tensor(shape=[None for _ in range(tensors.ndim)],
                                dtype=tensors.dtype)
    elif isinstance(tensors, (list, tuple)):
        return mutable([get_tensor_dynamic_input(t) for t in tensors])
    elif isinstance(tensors, (int, float, bool)):
        return mutable(tensors)
    else:
        raise ValueError


def enable_dynamic_shape(cell: nn.Cell, *cell_inputs, **kwargs):

    assert isinstance(cell, nn.Cell)

    fn_parameters = OrderedDict([
        (k, v)
        for k, v in inspect.signature(cell.construct).parameters.items()
    ])
    dynamic_inputs = []

    assert len(cell_inputs) + len(kwargs) <= len(fn_parameters)

    for i, (k, v) in enumerate(fn_parameters.items()):
        if k in kwargs:
            dynamic_input = get_tensor_dynamic_input(kwargs[k])
            dynamic_inputs.append(dynamic_input)
            continue

        if i < len(cell_inputs):
            dynamic_input = get_tensor_dynamic_input(cell_inputs[i])
            dynamic_inputs.append(dynamic_input)
        else:
            assert not isinstance(v, inspect.Parameter.empty)
            dynamic_input = get_tensor_dynamic_input(cell_inputs[i])
            dynamic_inputs.append(dynamic_input)

    cell.set_inputs(*dynamic_inputs)
