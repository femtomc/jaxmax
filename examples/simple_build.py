import numpy as np
from max import engine
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from typing import Any


def add_tensors(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    # 1. Build the graph
    input_type = TensorType(dtype=DType.float32, shape=(1,))
    with Graph("simple_add_graph", input_types=(input_type, input_type)) as graph:
        lhs, rhs = graph.inputs
        print(lhs, rhs)
        out = ops.add(lhs, rhs)
        print(out)
        graph.output(out)

    # 2. Create an inference session
    session = engine.InferenceSession()
    model = session.load(graph)

    # 3. Execute the graph
    ret = model.execute(a, b)[0]
    print("result:", ret)
    return ret


if __name__ == "__main__":
    input0 = np.array([1.0], dtype=np.float32)
    input1 = np.array([1.0], dtype=np.float32)
    add_tensors(input0, input1)
