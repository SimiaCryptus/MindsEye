/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.cudnn.f32;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.data.TensorList;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.CudaResource;
import jcuda.Pointer;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.cudnnOpTensor;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Product inputs layer.
 */
public class ProductInputsLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
  /**
   * From json product inputs layer.
   *
   * @param json the json
   * @return the product inputs layer
   */
  public static ProductInputsLayer fromJson(JsonObject json) {
    return new ProductInputsLayer(json);
  }
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param id the id
   */
  protected ProductInputsLayer(JsonObject id) {
    super(id);
  }
  
  /**
   * Instantiates a new Product inputs layer.
   */
  public ProductInputsLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    ((CudaExecutionContext) nncontext).initThread();
    assert inObj.length > 1;
    assert inObj.length < 3;
    int[] dimensions = inObj[0].getData().getDimensions();
    int length = inObj[0].getData().length();
    for (int i = 1; i < inObj.length; i++) {
      if (Tensor.dim(dimensions) != Tensor.dim(inObj[i].getData().getDimensions())) {
        throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(inObj[i].getData().getDimensions()));
      }
    }
    final CudaResource<cudnnOpTensorDescriptor> opDescriptor = CuDNN.newOpDescriptor(CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT);
    CudaResource<cudnnTensorDescriptor> sizeDescriptor = CuDNN.newTensorDescriptor(
      CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
    TensorList result = Arrays.stream(inObj).map(x -> x.getData()).reduce((l, r) -> {
      CudaPtr lPtr = CudaPtr.toDeviceAsFloat(((CudaExecutionContext) nncontext).getDeviceNumber(), l);
      CudaPtr rPtr = CudaPtr.toDeviceAsFloat(((CudaExecutionContext) nncontext).getDeviceNumber(), r);
      assert lPtr.size == rPtr.size;
      CudaPtr outputPtr = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), lPtr.size);
      CuDNN.handle(cudnnOpTensor(((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle, opDescriptor.getPtr(),
        Pointer.to(new float[]{1.0f}), sizeDescriptor.getPtr(), lPtr.getPtr(),
        Pointer.to(new float[]{1.0f}), sizeDescriptor.getPtr(), rPtr.getPtr(),
        Pointer.to(new float[]{0.0f}), sizeDescriptor.getPtr(), outputPtr.getPtr()));
      return CudaPtr.fromDeviceFloat(outputPtr, length, dimensions, ((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle);
    }).get();
    
    return new NNResult(result) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList delta) {
        ((CudaExecutionContext) nncontext).initThread();
        assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        for (int index = 0; index < inObj.length; index++) {
          final NNResult input = inObj[index];
          if (input.isAlive()) {
            int _index = index;
            input.accumulate(buffer, IntStream.range(0, inObj.length).mapToObj(i -> i == _index ? delta : inObj[i].getData()).reduce((l, r) -> {
              CudaPtr lPtr = CudaPtr.toDeviceAsFloat(((CudaExecutionContext) nncontext).getDeviceNumber(), l);
              CudaPtr rPtr = CudaPtr.toDeviceAsFloat(((CudaExecutionContext) nncontext).getDeviceNumber(), r);
              assert lPtr.size == rPtr.size;
              CudaPtr outputPtr = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), lPtr.size);
              CuDNN.handle(cudnnOpTensor(((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle, opDescriptor.getPtr(),
                Pointer.to(new float[]{1.0f}), sizeDescriptor.getPtr(), lPtr.getPtr(),
                Pointer.to(new float[]{1.0f}), sizeDescriptor.getPtr(), rPtr.getPtr(),
                Pointer.to(new float[]{0.0f}), sizeDescriptor.getPtr(), outputPtr.getPtr()));
              return CudaPtr.fromDeviceFloat(outputPtr, length, dimensions, ((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle);
            }).get());
          }
        }
      }
      
      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
