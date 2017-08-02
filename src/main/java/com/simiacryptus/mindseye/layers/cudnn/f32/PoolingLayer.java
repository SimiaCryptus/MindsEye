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
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.TensorList;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.DirectCuDNNLayer;
import com.simiacryptus.util.ml.Tensor;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;

import java.util.Arrays;
import java.util.List;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Pooling layer.
 */
public class PoolingLayer extends DirectCuDNNLayer {

  /**
   * From json pooling layer.
   *
   * @param json the json
   * @return the pooling layer
   */
  public static PoolingLayer fromJson(JsonObject json) {
    return new PoolingLayer(json);
  }

  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    return json;
  }

  /**
   * Instantiates a new Pooling layer.
   *
   * @param json the json
   */
  protected PoolingLayer(JsonObject json) {
    super(json);
  }

  /**
   * Instantiates a new Pooling layer.
   */
  public PoolingLayer() {
    super();
  }

  /**
   * The Pool dims.
   */
  final int poolDims = 2;
  /**
   * The Window size.
   */
  final int windowSize[] = { 2, 2 };
  /**
   * The Padding.
   */
  final int padding[] = { 0, 0 };
  /**
   * The Stride.
   */
  final int stride[] = { 2, 2 };
  /**
   * The Mode.
   */
  final int mode = CUDNN_POOLING_MAX;

  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    try {
      JCuda.cudaSetDevice(nncontext.getCudaDeviceId());
      //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
      final NNResult input = inObj[0];
      final TensorList batch = input.data;
      final int[] inputSize = batch.get(0).getDimensions();
      int length = batch.length();
      int inputDims = Tensor.dim(inputSize);
      CuDNN.CuDNNResource<cudnnPoolingDescriptor> poolingDesc = CuDNN.createPoolingDescriptor(
              mode, poolDims, windowSize, padding, stride);
      CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      int[] outputSize = new int[4];
      CuDNN.handle(cudnnGetPoolingNdForwardOutputDim(poolingDesc.getPtr(), inputDescriptor.getPtr(), 4, outputSize));
      assert(inputSize[2] == outputSize[1]);
      CuDNN.CuDNNResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outputSize[0], outputSize[1], outputSize[2], outputSize[3]);
      CuDNN.CuDNNPtr alpha = CuDNN.javaPtr(1.0f);
      CuDNN.CuDNNPtr beta = CuDNN.javaPtr(0.0f);
      CuDNN.CuDNNPtr inputData = toDeviceAsFloat(batch);
      CuDNN.CuDNNPtr outputData = CuDNN.alloc(Sizeof.FLOAT * Tensor.dim(outputSize));
      CuDNN.devicePool.with(device -> {
        CuDNN.handle(cudnnPoolingForward(device.cudnnHandle, poolingDesc.getPtr(),
                alpha.getPtr(),
                inputDescriptor.getPtr(), inputData.getPtr(),
                beta.getPtr(),
                outputDescriptor.getPtr(), outputData.getPtr()));
      });
      TensorList output = fromDeviceFloat(outputData, length, new int[]{outputSize[3], outputSize[2], outputSize[1]});
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          JCuda.cudaSetDevice(nncontext.getCudaDeviceId());
          assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          CuDNN.CuDNNPtr errorPtr = toDeviceAsFloat(error);
          if (input.isAlive()) {
            CuDNN.CuDNNPtr passbackBuffer = CuDNN.alloc(inputDims * Sizeof.FLOAT * length);
            CuDNN.devicePool.with(device -> {
              CuDNN.handle(cudnnPoolingBackward(device.cudnnHandle, poolingDesc.getPtr(),
                      alpha.getPtr(),
                      outputDescriptor.getPtr(), outputData.getPtr(),
                      outputDescriptor.getPtr(), errorPtr.getPtr(),
                      inputDescriptor.getPtr(), inputData.getPtr(),
                      beta.getPtr(),
                      inputDescriptor.getPtr(), passbackBuffer.getPtr()));
            });
            input.accumulate(buffer, fromDeviceFloat(passbackBuffer, length, inputSize));
          }
        }

        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new RuntimeException("Error",e);
    }
  }


  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
