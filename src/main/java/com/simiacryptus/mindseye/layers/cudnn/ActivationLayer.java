/*
 * Copyright (c) 2018 by Andrew Charneski.
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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.*;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer;
import jcuda.jcudnn.*;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The generic Activation layer, exposing the activation types provided by GpuSystem. This layer is stateless and is
 * determined by a univariate function, e.g. ReLU or Sigmoid.
 */
@SuppressWarnings("serial")
public class ActivationLayer extends NNLayer implements MultiPrecision<ActivationLayer> {
  /**
   * The Mode.
   */
  final int mode;
  private Precision precision = Precision.Double;
  
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param id the id
   */
  public ActivationLayer(final int id) {
    mode = id;
  }
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param json the json
   */
  protected ActivationLayer(final JsonObject json) {
    super(json);
    mode = json.getAsJsonPrimitive("mode").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param mode the mode
   */
  public ActivationLayer(final Mode mode) {
    this(mode.id);
  }
  
  /**
   * From json activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the activation layer
   */
  public static ActivationLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ActivationLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public NNLayer getCompatibilityLayer() {
    if (mode == Mode.SIGMOID.id) {
      return new SigmoidActivationLayer().setBalanced(false);
    }
    else if (mode == Mode.RELU.id) {
      return new ReLuActivationLayer();
    }
    else {
      throw new RuntimeException("Not Implemented");
    }
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    if (!GpuSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.getDimensions();
    final int[] outputSize = inputSize;
    final int length = batch.length();
    final int inputDims = Tensor.dim(inputSize);
    batch.addRef();
    try {
      CudaPtr outPtr = GpuSystem.eval(gpu -> {
        final CudaResource<cudnnTensorDescriptor> inputDescriptor = GpuSystem.newTensorDescriptor(
          precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
        final CudaResource<cudnnTensorDescriptor> outputDescriptor = GpuSystem.newTensorDescriptor(
          precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
        final CudaPtr inputData = CudaPtr.getCudaPtr(precision, batch);
        final CudaPtr outputData = CudaPtr.allocate(gpu.getDeviceNumber(), precision.size * 1l * inputDims * length, MemoryType.Managed, true);
        final CudaResource<cudnnActivationDescriptor> activationDesc = GpuSystem.newActivationDescriptor(mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
        try {
          GpuSystem.handle(CuDNNHandle.cudnnActivationForward(gpu.getHandle(), activationDesc.getPtr(),
                                                              precision.getPointer(1.0),
                                                              inputDescriptor.getPtr(), inputData.getPtr(),
                                                              precision.getPointer(0.0),
                                                              outputDescriptor.getPtr(), outputData.getPtr()));
        } catch (final Throwable e) {
          throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
        }
        gpu.registerForCleanup(inputData, inputDescriptor, outputDescriptor, activationDesc);
        return outputData;
        //assert output.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      });
      return new NNResult(GpuTensorList.create(outPtr, length, outputSize, precision), (final DeltaSet<NNLayer> buffer, final TensorList error) -> {
        if (input.isAlive()) {
          final TensorList data = GpuSystem.eval(gpu -> {
            //assert (error.length() == batch.length());
            //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
            final CudaPtr inputData = CudaPtr.getCudaPtr(precision, batch);
            final CudaResource<cudnnTensorDescriptor> inputDescriptor = GpuSystem.newTensorDescriptor(
              precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
            final CudaPtr errorPtr = CudaPtr.getCudaPtr(precision, error);
            final CudaPtr passbackBuffer = CudaPtr.allocate(gpu.getDeviceNumber(), inputDims * 1l * precision.size * length, MemoryType.Managed, true);
            final CudaResource<cudnnActivationDescriptor> activationDesc = GpuSystem.newActivationDescriptor(mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
            try {
              GpuSystem.handle(CuDNNHandle.cudnnActivationBackward(gpu.getHandle(), activationDesc.getPtr(),
                                                                   precision.getPointer(1.0),
                                                                   inputDescriptor.getPtr(), outPtr.getPtr(),
                                                                   inputDescriptor.getPtr(), errorPtr.getPtr(),
                                                                   inputDescriptor.getPtr(), inputData.getPtr(),
                                                                   precision.getPointer(0.0),
                                                                   inputDescriptor.getPtr(), passbackBuffer.getPtr()));
            } catch (final Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
            }
            gpu.registerForCleanup(inputData, errorPtr, inputDescriptor, activationDesc);
            return GpuTensorList.wrap(passbackBuffer, length, inputSize, precision);
          });
          input.accumulate(buffer, data);
          data.freeRef();
        }
      }) {
  
        @Override
        protected void _free() {
          batch.freeRef();
          outPtr.freeRef();
          Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
        }
  
  
        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (final Throwable e) {
      throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
    }
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public ActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  
  /**
   * The enum Mode.
   */
  public enum Mode {
    /**
     * Relu mode.
     */
    RELU(cudnnActivationMode.CUDNN_ACTIVATION_RELU),
    /**
     * Sigmoid mode.
     */
    SIGMOID(cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID);
    /**
     * The Id.
     */
    public final int id;
  
    Mode(final int id) {
      this.id = id;
    }
  }
  
}
