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
import com.simiacryptus.mindseye.lang.ComponentException;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaDevice;
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.ImgPixelSoftmaxLayer;
import jcuda.jcudnn.cudnnSoftmaxAlgorithm;
import jcuda.jcudnn.cudnnSoftmaxMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The classic "softmax" layer. All outputs will sum to 1 and be proportional to the log of the input.
 */
@SuppressWarnings("serial")
public class SoftmaxActivationLayer extends LayerBase implements MultiPrecision<SoftmaxActivationLayer> {
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);
  private SoftmaxAlgorithm algorithm = SoftmaxAlgorithm.ACCURATE;
  private SoftmaxMode mode = SoftmaxMode.INSTANCE;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Activation layer.
   */
  public SoftmaxActivationLayer() {
  
  }
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param json the json
   */
  protected SoftmaxActivationLayer(@Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
    algorithm = SoftmaxAlgorithm.valueOf(json.get("algorithm").getAsString());
    mode = SoftmaxMode.valueOf(json.get("mode").getAsString());
  }
  
  /**
   * From json activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the activation layer
   */
  public static SoftmaxActivationLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new SoftmaxActivationLayer(json);
  }
  
  /**
   * Gets algorithm.
   *
   * @return the algorithm
   */
  public SoftmaxAlgorithm getAlgorithm() {
    return algorithm;
  }
  
  /**
   * Sets algorithm.
   *
   * @param algorithm the algorithm
   * @return the algorithm
   */
  public SoftmaxActivationLayer setAlgorithm(SoftmaxAlgorithm algorithm) {
    this.algorithm = algorithm;
    return this;
  }
  
  /**
   * Gets mode.
   *
   * @return the mode
   */
  public SoftmaxMode getMode() {
    return mode;
  }
  
  /**
   * Sets mode.
   *
   * @param mode the mode
   * @return the mode
   */
  public SoftmaxActivationLayer setMode(SoftmaxMode mode) {
    this.mode = mode;
    return this;
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    assert algorithm != SoftmaxAlgorithm.LOG;
    if (mode == SoftmaxMode.CHANNEL) return this.as(ImgPixelSoftmaxLayer.class);
    return this.as(com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer.class);
  }
  
  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    final Result inputResult = inObj[0];
    final TensorList inputData = inputResult.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    @Nonnull final int[] outputSize = inputSize;
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    try {
      final CudaTensor outPtr = CudaSystem.run(gpu -> {
        @Nullable CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
        final CudaTensor outputTensor;
        if (1 == inputData.currentRefCount() && 1 == inputTensor.currentRefCount()) {
          outputTensor = inputTensor;
          outputTensor.addRef();
        }
        else {
          @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision,
            length, inputSize[2], inputSize[1], inputSize[0],
            inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0], inputSize[0], 1);
          @Nonnull final CudaMemory outputData =
            gpu.allocate(precision.size * 1l * inputDims * length, MemoryType.Managed.normalize(), true);
          outputTensor = CudaTensor.wrap(outputData, outputDescriptor, precision);
        }
        try {
          CudaMemory inputMemory = inputTensor.getMemory(gpu);
          CudaMemory outputMemory = outputTensor.getMemory(gpu);
          CudaSystem.handle(gpu.cudnnSoftmaxForward(algorithm.code, mode.code,
            precision.getPointer(1.0), inputTensor.descriptor.getPtr(), inputMemory.getPtr(),
            precision.getPointer(0.0), outputTensor.descriptor.getPtr(), outputMemory.getPtr()
          ));
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          inputMemory.dirty();
          outputMemory.dirty();
          outputMemory.freeRef();
          inputMemory.freeRef();
          return outputTensor;
        } catch (@Nonnull final Throwable e) {
          throw new ComponentException("Error apply " + Arrays.toString(inputSize), e);
        } finally {
          inputTensor.freeRef();
        }
      }, inputData);
      return new Result(CudaTensorList.create(outPtr, length, outputSize, precision),
        (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
          if (inputResult.isAlive()) {
            final TensorList data = CudaSystem.run(gpu -> {
              final CudaTensor result1;
              synchronized (gpu) {result1 = gpu.getTensor(inputData, precision, MemoryType.Device, true);}
              @Nullable CudaTensor inputTensor = result1;
              final CudaTensor result;
              synchronized (gpu) {result = gpu.getTensor(delta, precision, MemoryType.Device, true);}
              @Nullable CudaTensor deltaTensor = result;
              outPtr.addRef();
              CudaTensor localOut = outPtr.getDenseAndFree(gpu);
              delta.freeRef();
              CudaTensor passbackTensor;
              passbackTensor = CudaTensor.wrap(
                gpu.allocate((long) Tensor.length(inputSize) * length * precision.size, MemoryType.Managed.normalize(), false),
                gpu.newTensorDescriptor(precision,
                  delta.length(), inputSize[2], inputSize[1], inputSize[0],
                  inputSize[2] * inputSize[1] * inputSize[0],
                  inputSize[1] * inputSize[0],
                  inputSize[0],
                  1), precision);
  
              try {
                CudaMemory localOutMemory = localOut.getMemory(gpu);
                CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                CudaMemory inputMemory = inputTensor.getMemory(gpu);
                CudaMemory passbackMemory = passbackTensor.getMemory(gpu);
    
                CudaSystem.handle(gpu.cudnnSoftmaxBackward(algorithm.code, mode.code,
                  precision.getPointer(1.0), localOut.descriptor.getPtr(), localOutMemory.getPtr(),
                  deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                  precision.getPointer(0.0), passbackTensor.descriptor.getPtr(), passbackMemory.getPtr()
                ));
                localOutMemory.dirty();
                deltaTensorMemory.dirty();
                passbackMemory.dirty();
  
                localOutMemory.freeRef();
                deltaTensorMemory.freeRef();
                inputMemory.freeRef();
                passbackMemory.freeRef();
              } catch (@Nonnull final Throwable e) {
                throw new ComponentException("Error apply " + Arrays.toString(inputSize), e);
              } finally {
                localOut.freeRef();
                inputTensor.freeRef();
                deltaTensor.freeRef();
              }
              return CudaTensorList.wrap(passbackTensor, length, inputSize, precision);
            }, delta);
            inputResult.accumulate(buffer, data);
          }
          else {
            delta.freeRef();
          }
        }) {
  
        @Override
        public final void accumulate(DeltaSet<Layer> buffer, TensorList delta) {
          getAccumulator().accept(buffer, delta);
        }
  
        @Override
        protected void _free() {
          inputData.freeRef();
          outPtr.freeRef();
          inputResult.freeRef();
        }
  
  
        @Override
        public boolean isAlive() {
          return inputResult.isAlive() || !isFrozen();
        }
      };
    } catch (@Nonnull final Throwable e) {
      throw new ComponentException("Error apply image res " + Arrays.toString(inputSize), e);
    }
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.addProperty("algorithm", algorithm.name());
    json.addProperty("mode", mode.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public SoftmaxActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  /**
   * The enum Softmax algorithm.
   */
  public enum SoftmaxAlgorithm {
    /**
     * Fast softmax algorithm.
     */
    FAST(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_FAST),
    /**
     * Accurate softmax algorithm.
     */
    ACCURATE(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE),
    /**
     * Log softmax algorithm.
     */
    LOG(cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_LOG);
  
    /**
     * The Code.
     */
    public final int code;
    
    SoftmaxAlgorithm(final int code) {
      this.code = code;
    }
  }
  
  /**
   * The enum Softmax mode.
   */
  public enum SoftmaxMode {
    /**
     * Channel softmax mode.
     */
    CHANNEL(cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL),
    /**
     * Instance softmax mode.
     */
    INSTANCE(cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_INSTANCE);
  
    /**
     * The Code.
     */
    public final int code;
    
    SoftmaxMode(final int code) {
      this.code = code;
    }
  }
  
}
