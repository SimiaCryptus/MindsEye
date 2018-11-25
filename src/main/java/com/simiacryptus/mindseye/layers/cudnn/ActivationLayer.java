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
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnActivationMode;
import jcuda.jcudnn.cudnnNanPropagation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * The generic Activation key, exposing the activation types provided by CudaSystem. This key is stateless and is
 * determined by a univariate function, e.g. ReLU or Sigmoid.
 */
@SuppressWarnings("serial")
public class ActivationLayer extends LayerBase implements MultiPrecision<ActivationLayer> {
  private static final Logger logger = LoggerFactory.getLogger(ActivationLayer.class);

  /**
   * The Mode.
   */
  final int mode;
  private double alpha = 1.0;
  private Precision precision = Precision.Double;


  /**
   * Instantiates a new Activation key.
   *
   * @param id the id
   */
  public ActivationLayer(final int id) {
    mode = id;
  }

  /**
   * Instantiates a new Activation key.
   *
   * @param json the json
   */
  protected ActivationLayer(@Nonnull final JsonObject json) {
    super(json);
    mode = json.getAsJsonPrimitive("mode").getAsInt();
    setAlpha(json.getAsJsonPrimitive("alpha").getAsDouble());
    precision = Precision.valueOf(json.get("precision").getAsString());
  }

  /**
   * Instantiates a new Activation key.
   *
   * @param mode the mode
   */
  public ActivationLayer(@Nonnull final Mode mode) {
    this(mode.id);
  }

  /**
   * From json activation key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the activation key
   */
  public static ActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ActivationLayer(json);
  }

  /**
   * Same strides boolean.
   *
   * @param a the a
   * @param b the b
   * @return the boolean
   */
  public static boolean sameStrides(final CudaDevice.CudaTensorDescriptor a, final CudaDevice.CudaTensorDescriptor b) {
    if (a.nStride != b.nStride) return false;
    if (a.cStride != b.cStride) return false;
    if (a.hStride != b.hStride) return false;
    return a.wStride == b.wStride;
  }

  /**
   * Gets compatibility key.
   *
   * @return the compatibility key
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    if (mode == Mode.SIGMOID.id) {
      return new SigmoidActivationLayer().setBalanced(false);
    } else if (mode == Mode.RELU.id) {
      return new ReLuActivationLayer();
    } else {
      throw new RuntimeException("Not Implemented");
    }
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final Result inputResult = inObj[0];
    final TensorList inputData = inputResult.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    @Nonnull final int[] outputSize = inputSize;
    final int length = inputData.length();
    final int inputDims = Tensor.length(inputSize);
    try {
      final CudaTensor outPtr = CudaSystem.run(gpu -> {
        @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
        final CudaTensor outputTensor;
        if (1 == inputData.currentRefCount() && 1 == inputTensor.currentRefCount() && (!inputResult.isAlive() || mode == Mode.RELU.id)) {
          inputTensor.addRef();
          outputTensor = inputTensor;
        } else {
          @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision,
              length, inputSize[2], inputSize[1], inputSize[0],
              inputSize[2] * inputSize[1] * inputSize[0], inputSize[1] * inputSize[0], inputSize[0], 1);
          @Nonnull final CudaMemory outputData =
              gpu.allocate((long) precision.size * inputDims * length, MemoryType.Managed.normalize(), true);
          outputTensor = CudaTensor.wrap(outputData, outputDescriptor, precision);
        }

        @Nonnull final CudaResource<cudnnActivationDescriptor> activationDesc = gpu.newActivationDescriptor(mode, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
        try {
          CudaMemory memory = inputTensor.getMemory(gpu);
          CudaMemory tensorMemory = outputTensor.getMemory(gpu);
          CudaSystem.handle(gpu.cudnnActivationForward(activationDesc.getPtr(),
              precision.getPointer(getAlpha()), inputTensor.descriptor.getPtr(), memory.getPtr(),
              precision.getPointer(0.0), outputTensor.descriptor.getPtr(), tensorMemory.getPtr()));
          assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
          memory.dirty();
          tensorMemory.dirty();
          tensorMemory.freeRef();
          memory.freeRef();
          return outputTensor;
        } catch (@Nonnull final Throwable e) {
          throw new ComponentException("Error apply " + Arrays.toString(inputSize), e);
        } finally {
          activationDesc.freeRef();
          inputTensor.freeRef();
        }
      }, inputData);
      return new Result(CudaTensorList.create(outPtr, length, outputSize, precision),
          (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
            if (inputResult.isAlive()) {
              final TensorList data = CudaSystem.run(gpu -> {
                @Nullable CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, true);
                @Nullable CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, true);
                assert length == delta.length();
                CudaTensor localOut = outPtr.getDense(gpu);
                delta.freeRef();
                CudaTensor passbackTensor;
//            if (sameStrides(deltaTensor.descriptor, inputTensor.descriptor)) {
//              passbackTensor = deltaTensor;
//              passbackTensor.addRef();
//            }
//            else {
//              passbackTensor = deltaTensor.getDense(gpu);
//              inputTensor = inputTensor.getDenseAndFree(gpu);
//            }
                passbackTensor = CudaTensor.wrap(
                    gpu.allocate((long) Tensor.length(inputSize) * length * precision.size, MemoryType.Managed.normalize(), false),
                    gpu.newTensorDescriptor(precision,
                        length, inputSize[2], inputSize[1], inputSize[0],
                        inputSize[2] * inputSize[1] * inputSize[0],
                        inputSize[1] * inputSize[0],
                        inputSize[0],
                        1), precision);

                @Nonnull final CudaResource<cudnnActivationDescriptor> activationDesc = gpu.newActivationDescriptor(mode,
                    cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN, 0);
                try {
                  CudaMemory localOutMemory = localOut.getMemory(gpu);
                  CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
                  CudaMemory inputTensorMemory = inputTensor.getMemory(gpu);
                  CudaMemory passbackTensorMemory = passbackTensor.getMemory(gpu);
                  CudaSystem.handle(gpu.cudnnActivationBackward(activationDesc.getPtr(),
                      precision.getPointer(getAlpha()),
                      localOut.descriptor.getPtr(), localOutMemory.getPtr(),
                      deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
                      inputTensor.descriptor.getPtr(), inputTensorMemory.getPtr(),
                      precision.getPointer(0.0),
                      passbackTensor.descriptor.getPtr(), passbackTensorMemory.getPtr()));
                  assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
                  localOutMemory.dirty();
                  deltaTensorMemory.dirty();
                  inputTensorMemory.dirty();
                  passbackTensorMemory.dirty();
                  localOutMemory.freeRef();
                  deltaTensorMemory.freeRef();
                  inputTensorMemory.freeRef();
                  passbackTensorMemory.freeRef();
                } catch (@Nonnull final Throwable e) {
                  throw new ComponentException("Error apply " + Arrays.toString(inputSize), e);
                } finally {
                  localOut.freeRef();
                  inputTensor.freeRef();
                  deltaTensor.freeRef();
                  activationDesc.freeRef();
                }
                return CudaTensorList.wrap(passbackTensor, length, inputSize, precision);
              }, delta);
              inputResult.accumulate(buffer, data);
            } else {
              delta.freeRef();
            }
          }) {

        @Override
        public final void accumulate(DeltaSet<UUID> buffer, TensorList delta) {
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
      throw new ComponentException("Error apply png res " + Arrays.toString(inputSize), e);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("alpha", getAlpha());
    json.addProperty("mode", mode);
    json.addProperty("precision", precision.name());
    return json;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public ActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

  /**
   * Gets alpha.
   *
   * @return the alpha
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Sets alpha.
   *
   * @param alpha the alpha
   * @return the alpha
   */
  public ActivationLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
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
