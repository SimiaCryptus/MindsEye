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

package com.simiacryptus.mindseye.layers.cudnn.f64;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.CudaResource;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;

import static jcuda.jcudnn.JCudnn.cudnnTransformTensor;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;

/**
 * The type Img band bias layer.
 */
public class ImgConcatLayer extends NNLayer {
  
  /**
   * From json img band bias layer.
   *
   * @param json the json
   * @return the img band bias layer
   */
  public static ImgConcatLayer fromJson(JsonObject json) {
    return new ImgConcatLayer(json);
  }

  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    return json;
  }
  
  /**
   * Instantiates a new Img band bias layer.
   *
   * @param json the json
   */
  protected ImgConcatLayer(JsonObject json) {
    super(json);
  }
  
  /**
   * Instantiates a new Img band bias layer.
   */
  public ImgConcatLayer() {
  }

  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    assert 3 == inObj[0].getData().getDimensions().length;
    int[] dimOut = Arrays.copyOf(inObj[0].getData().getDimensions(), 3);
    int length = inObj[0].getData().length();
    assert Arrays.stream(inObj).allMatch(x -> 3 == x.getData().getDimensions().length && x.getData().getDimensions()[0] == dimOut[0] && x.getData().getDimensions()[1] == dimOut[1] && x.getData().length() == length);
    dimOut[2] = Arrays.stream(inObj).mapToInt(x -> x.getData().getDimensions()[2]).sum();
    ((CudaExecutionContext) nncontext).initThread();
    CudaPtr outputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), length * dimOut[2] * dimOut[1] * dimOut[0] * Sizeof.DOUBLE);
    int bandOffset = 0;
    for (int i = 0; i < inObj.length; i++) {
      TensorList data = inObj[i].getData();
      int[] dimensions = data.getDimensions();
      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_DOUBLE, length, dimensions[2], dimensions[1], dimensions[0],
        dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0], dimensions[0], 1);
      CudaResource<cudnnTensorDescriptor> viewDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_DOUBLE, length, dimensions[2], dimensions[1], dimensions[0],
        dimOut[2] * dimOut[1] * dimOut[0], dimOut[1] * dimOut[0], dimOut[0], 1);
      CudaPtr cudaPtr = CudaPtr.toDeviceAsDouble(((CudaExecutionContext) nncontext).getDeviceNumber(), data);
      cudnnTransformTensor(((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle,
        Pointer.to(new double[]{1.0}), inputDescriptor.getPtr(), cudaPtr.getPtr(),
        Pointer.to(new double[]{0.0}), viewDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(dimensions[1] * dimensions[0] * bandOffset * Sizeof.DOUBLE)
      );
      bandOffset += dimensions[2];
    }
    TensorList outputData = CudaPtr.fromDeviceDouble(outputBuffer, length, dimOut, ((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle);
    //assert outputData.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    return new NNResult(outputData) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList error) {
        outputBuffer.finalize();
        ((CudaExecutionContext) nncontext).initThread();
        assert (error.length() == inObj[0].getData().length());
        //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
        CudaPtr errorPtr = CudaPtr.toDeviceAsDouble(((CudaExecutionContext) nncontext).getDeviceNumber(), error);
        int bandOffset = 0;
        for (int i = 0; i < inObj.length; i++) {
          NNResult input = inObj[i];
          int[] dimensions = input.getData().getDimensions();
          if (input.isAlive()) {
            int _bandOffset = bandOffset;
            CudaPtr passbackBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), length * dimensions[2] * dimensions[1] * dimensions[0] * Sizeof.DOUBLE);
            CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_DOUBLE, length, dimensions[2], dimensions[1], dimensions[0],
              dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0], dimensions[0], 1);
            CudaResource<cudnnTensorDescriptor> viewDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_DOUBLE, length, dimensions[2], dimensions[1], dimensions[0],
              dimOut[2] * dimOut[1] * dimOut[0], dimOut[1] * dimOut[0], dimOut[0], 1);
            cudnnTransformTensor(((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle,
              Pointer.to(new double[]{1.0}), viewDescriptor.getPtr(), errorPtr.getPtr().withByteOffset(dimensions[1] * dimensions[0] * _bandOffset * Sizeof.DOUBLE),
              Pointer.to(new double[]{0.0}), inputDescriptor.getPtr(), passbackBuffer.getPtr()
            );
            TensorList passbackTensorList = CudaPtr.fromDeviceDouble(passbackBuffer, length, dimensions, ((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle);
            //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
            input.accumulate(buffer, passbackTensorList);
            passbackBuffer.finalize();
          }
          bandOffset += dimensions[2];
        }
      }

      @Override
      public boolean isAlive() {
        return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
      }
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
