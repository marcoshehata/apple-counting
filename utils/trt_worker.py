import os
import tensorrt as trt
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart
import numpy as np

class TRTModel:
    """
    High-performance TensorRT engine wrapper for NVIDIA Jetson.
    Uses cuda-python for direct memory management to avoid heavy dependencies like torch-cuda.
    """
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
            
        print(f"[*] Loading TensorRT Engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if not self.engine:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
            
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        # TRT 10+ IO Tensor Management
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.engine.get_tensor_shape(name)
            
            # Calculate size in bytes
            size = np.prod(shape) * dtype.itemsize
            
            # Allocate device memory
            err, ptr = cudart.cudaMalloc(size)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA Malloc failed for {name}: {err}")
                
            self.allocations.append(ptr)
            
            binding = {
                'index': i,
                'name': name,
                'dtype': dtype,
                'shape': list(shape),
                'size': size,
                'ptr': int(ptr)
            }
            
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            
            # Set tensor address for execution context
            self.context.set_tensor_address(name, int(ptr))

    def infer(self, input_data: list) -> list:
        """
        Runs synchronous inference.
        Args:
            input_data: List of numpy arrays matching input bindings.
        Returns:
            List of numpy arrays for all output bindings.
        """
        # Host to Device
        for i, data in enumerate(input_data):
            if i >= len(self.inputs): break
            # Ensure data is contiguous and correct type
            data = np.ascontiguousarray(data).astype(self.inputs[i]['dtype'])
            err, = cudart.cudaMemcpy(self.inputs[i]['ptr'], data.ctypes.data, self.inputs[i]['size'], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA H2D Memcpy failed: {err}")
        
        # Execute (Async V3 is standard in TRT 10+, but we can use stream 0 for synchronous behavior)
        self.context.execute_async_v3(0)
        
        # Device to Host
        output_results = []
        for binding in self.outputs:
            data = np.empty(binding['shape'], dtype=binding['dtype'])
            err, = cudart.cudaMemcpy(data.ctypes.data, binding['ptr'], binding['size'], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA D2H Memcpy failed: {err}")
            output_results.append(data)
            
        return output_results

    def __del__(self):
        """Cleanup CUDA allocations."""
        if hasattr(self, 'allocations'):
            for ptr in self.allocations:
                cudart.cudaFree(ptr)
