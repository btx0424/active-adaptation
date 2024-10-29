import onnxruntime as ort
import json
import numpy as np
from typing import Dict


class ONNXModule:
    
    def __init__(self, path: str):

        self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        with open(path.replace(".onnx", ".json"), "r") as f:
            self.meta = json.load(f)
        self.in_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["in_keys"]]
        self.out_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["out_keys"]]
    
    def __call__(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        args = {
            inp.name: input[key]
            for inp, key in zip(self.ort_session.get_inputs(), self.in_keys)
            if key in input
        }
        outputs = self.ort_session.run(None, args)
        outputs = {k: v for k, v in zip(self.out_keys, outputs)}
        return outputs
    
