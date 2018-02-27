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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The generic Activation layer, exposing the activation types provided by CudaSystem. This layer is stateless and is
 * determined by a univariate function, e.g. ReLU or Sigmoid.
 */
@SuppressWarnings("serial")
public class ActivationLayer extends LayerBase implements MultiPrecision<ActivationLayer> {
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
  protected ActivationLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    mode = json.getAsJsonPrimitive("mode").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param mode the mode
   */
  public ActivationLayer(@javax.annotation.Nonnull final Mode mode) {
    this(mode.id);
  }
  
  /**
   * From json activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the activation layer
   */
  public static ActivationLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ActivationLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public Layer getCompatibilityLayer() {
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
  
  @Nullable
  @Override
  public Result evalAndFree(@javax.annotation.Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final Result inputResult = inObj[0];
    final TensorList inputData = inputResult.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    @Nonnull final int[] outputSize = inputSize;
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    try {
      CudaTensor outPtr = CudaSystem.eval(gpu -> {
        @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> inputDescriptor = gpu.newTensorDescriptor(
          precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
        @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> outputDescriptor = gpu.newTensorDescriptor(
          precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
        @Nullable final CudaTensor inputPtr = gpu.getTensor(inputData, precision, MemoryType.Device);
        @javax.annotation.Nonnull final CudaMemory outputData;
        if (1 == inputData.currentRefCount() && 1 == inputPtr.currentRefCount() && (!inputResult.isAlive() || mode == Mode.RELU.id)) {
          outputData = inputPtr.memory;
          outputData.addRef();
        }
        else {
          outputData = gpu.allocate(precision.size * 1l * inputDims * length, MemoryType.Managed, true);
        }
        @javax.annotation.Nonnull final CudaResource<cudnnActivationDescriptor> activationDesc = gpu.newActivationDescriptor(mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
        try {
          CudaSystem.handle(gpu.cudnnActivationForward(activationDesc.getPtr(),
            precision.getPointer(1.0),
            inputDescriptor.getPtr(), inputPtr.memory.getPtr(),
            precision.getPointer(0.0),
            outputDescriptor.getPtr(), outputData.getPtr()));
        } catch (@javax.annotation.Nonnull final Throwable e) {
          throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
        }
        CudaTensor cudaTensor = CudaTensor.wrap(outputData, outputDescriptor);
        inputPtr.freeRef();
        inputDescriptor.freeRef();
        activationDesc.freeRef();
        return cudaTensor;
        //assert output.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      });
      return new Result(CudaTensorList.create(outPtr, length, outputSize, precision), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
        if (inputResult.isAlive()) {
          final TensorList data = CudaSystem.eval(gpu -> {
            //assert (error.length() == batch.length());
            //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
            @Nullable final CudaTensor inputPtr = gpu.getTensor(inputData, precision, MemoryType.Device);
            @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> inputDescriptor = gpu.newTensorDescriptor(
              precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
            @Nullable final CudaTensor deltaPtr = gpu.getTensor(delta, precision, MemoryType.Device);
            @javax.annotation.Nonnull final CudaMemory passbackBuffer;
            if (1 == delta.currentRefCount() && 1 == deltaPtr.currentRefCount()) {
              passbackBuffer = deltaPtr.memory;
              passbackBuffer.addRef();
              delta.freeRef();
            }
            else {
              passbackBuffer = gpu.allocate(inputDims * 1l * precision.size * length, MemoryType.Managed, true);
            }
            @javax.annotation.Nonnull final CudaResource<cudnnActivationDescriptor> activationDesc = gpu.newActivationDescriptor(mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
            try {
              CudaSystem.handle(gpu.cudnnActivationBackward(activationDesc.getPtr(),
                precision.getPointer(1.0),
                outPtr.descriptor.getPtr(), outPtr.memory.getPtr(),
                inputDescriptor.getPtr(), deltaPtr.memory.getPtr(),
                inputDescriptor.getPtr(), inputPtr.memory.getPtr(),
                precision.getPointer(0.0),
                inputDescriptor.getPtr(), passbackBuffer.getPtr()));
            } catch (@javax.annotation.Nonnull final Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
            }
            CudaTensor cudaTensor = new CudaTensor(passbackBuffer, inputDescriptor);
            inputPtr.freeRef();
            deltaPtr.freeRef();
            inputDescriptor.freeRef();
            activationDesc.freeRef();
            return CudaTensorList.wrap(cudaTensor, length, inputSize, precision);
          });
          inputResult.accumulate(buffer, data);
        }
      }) {
        
        @Override
        protected void _free() {
          inputData.freeRef();
          outPtr.freeRef();
          Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
        }
        
        
        @Override
        public boolean isAlive() {
          return inputResult.isAlive() || !isFrozen();
        }
      };
    } catch (@javax.annotation.Nonnull final Throwable e) {
      throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
    }
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public ActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @javax.annotation.Nonnull
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
