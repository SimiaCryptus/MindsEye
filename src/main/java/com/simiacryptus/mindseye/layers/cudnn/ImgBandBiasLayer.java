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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaDevice;
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory;
import com.simiacryptus.mindseye.lang.cudnn.CudaResource;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.ProductInputsLayer;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Stream;

/**
 * This layer multiplies together the inputs, element-by-element. It can be used to implement integer-power activation
 * layers, such as the square needed in MeanSqLossLayer.
 */
@SuppressWarnings("serial")
public class ImgBandBiasLayer extends LayerBase implements MultiPrecision<ImgBandBiasLayer> {
  
  private Precision precision = Precision.Double;
  private Tensor bias;
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param bands the bands
   */
  public ImgBandBiasLayer(int bands) {
    this.bias = new Tensor(1, 1, bands);
  }
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param id the id
   * @param rs the rs
   */
  protected ImgBandBiasLayer(@Nonnull final JsonObject id, final Map<String, byte[]> rs) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
    this.bias = Tensor.fromJson(id.get("bias"), rs);
  }
  
  /**
   * From json product inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the product inputs layer
   */
  public static ImgBandBiasLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgBandBiasLayer(json, rs);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(ProductInputsLayer.class);
  }
  
  
  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    if (inObj.length != 1) {
      throw new IllegalArgumentException("inObj.length=" + inObj.length);
    }
    Result input = inObj[0];
    final TensorList leftData = input.getData();
    @Nonnull final int[] inputDimensions = leftData.getDimensions();
    final int length = leftData.length();
    if (3 != inputDimensions.length) {
      throw new IllegalArgumentException("dimensions=" + Arrays.toString(inputDimensions));
    }
    leftData.addRef();
    input.addRef();
//   assert !right.isAlive();
    return new Result(CudaSystem.eval(gpu -> {
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
      @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
        inputDimensions[2], inputDimensions[1], inputDimensions[0],
        inputDimensions[2] * inputDimensions[1] * inputDimensions[0],
        inputDimensions[1] * inputDimensions[0],
        inputDimensions[0],
        1);
      @Nullable final CudaTensor inputTensor = gpu.getTensor(leftData, precision, MemoryType.Device, false);
      CudaMemory biasMem = gpu.allocate(bias.size() * precision.size, MemoryType.Device, true).write(precision, bias.getData());
      int[] biasDim = bias.getDimensions();
      CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision, 1, biasDim[2], biasDim[1], biasDim[0],
        biasDim[2] * biasDim[1] * biasDim[0], biasDim[1] * biasDim[0], biasDim[0], 1);
      //assert lPtr.size == rPtr.size;
      @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length, MemoryType.Managed.normalize(), true);
      CudaMemory inputMemory = inputTensor.getMemory(gpu);
      CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(),
        precision.getPointer(1.0), inputTensor.descriptor.getPtr(), inputMemory.getPtr(),
        precision.getPointer(1.0), biasDescriptor.getPtr(), biasMem.getPtr(),
        precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
      inputMemory.dirty();
      biasMem.dirty();
      outputPtr.dirty();
      inputMemory.freeRef();
      biasMem.freeRef();
      biasDescriptor.freeRef();
      inputTensor.freeRef();
      opDescriptor.freeRef();
      CudaTensor cudaTensor = CudaTensor.wrap(outputPtr, outputDescriptor, precision);
      return CudaTensorList.wrap(cudaTensor, length, inputDimensions, precision);
    }, leftData), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (!isFrozen()) {
        @Nonnull double[] biasDelta = CudaSystem.eval(gpu -> {
          @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
          @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, false);
  
          CudaMemory biasMem = gpu.allocate(bias.size() * precision.size, MemoryType.Device, true).write(precision, bias.getData());
          int[] biasDim = bias.getDimensions();
          CudaDevice.CudaTensorDescriptor biasDescriptor = gpu.newTensorDescriptor(precision,
            1, biasDim[2], biasDim[1], biasDim[0],
            biasDim[2] * biasDim[1] * biasDim[0], biasDim[1] * biasDim[0], biasDim[0], 1);
          CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
          gpu.cudnnConvolutionBackwardBias(precision.getPointer(1.0), deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
            precision.getPointer(0.0), biasDescriptor.getPtr(), biasMem.getPtr());
          biasMem.dirty();
          double[] biasV = new double[bias.length()];
          biasMem.synchronize();
          biasMem.read(precision, biasV);
          Stream.<ReferenceCounting>of(biasMem, deltaTensorMemory, deltaTensor, opDescriptor, biasDescriptor).forEach(ReferenceCounting::freeRef);
          return biasV;
        }, delta);
        buffer.get(ImgBandBiasLayer.this, bias).addInPlace(biasDelta).freeRef();
      }
      if (input.isAlive()) {
        input.accumulate(buffer, delta);
      }
      else {
        delta.freeRef();
      }
    }) {
  
      @Override
      protected boolean autofree() {
        return false;
      }
      
      @Override
      protected void _free() {
        leftData.freeRef();
        input.freeRef();
      }
  
  
      @Override
      public boolean isAlive() {
        for (@Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
  
    };
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    json.add("bias", bias.toJson(resources, dataSerializer));
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public ImgBandBiasLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(bias.getData());
  }
  
  /**
   * Add weights img band bias layer.
   *
   * @param f the f
   * @return the img band bias layer
   */
  @Nonnull
  public ImgBandBiasLayer addWeights(@Nonnull final DoubleSupplier f) {
    Util.add(f, getBias());
    return this;
  }
  
  /**
   * Add weights img band bias layer.
   *
   * @param f the f
   * @return the img band bias layer
   */
  @Nonnull
  public ImgBandBiasLayer setWeights(@Nonnull final IntToDoubleFunction f) {
    bias.setByCoord(c -> f.applyAsDouble(c.getIndex()));
    return this;
  }
  
  /**
   * Sets weights log.
   *
   * @param value the value
   * @return the weights log
   */
  @Nonnull
  public ImgBandBiasLayer setWeightsLog(final double value) {
    bias.setByCoord(c -> (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value));
    return this;
  }
  
  /**
   * Sets and free.
   *
   * @param tensor the tensor
   * @return the and free
   */
  public ImgBandBiasLayer setAndFree(final Tensor tensor) {
    set(tensor);
    tensor.freeRef();
    return this;
  }
  
  /**
   * Set img band bias layer.
   *
   * @param tensor the tensor
   * @return the img band bias layer
   */
  public ImgBandBiasLayer set(final Tensor tensor) {
    bias.set(tensor);
    return this;
  }
  
  /**
   * Get bias double [ ].
   *
   * @return the double [ ]
   */
  public double[] getBias() {
    return bias.getData();
  }
  
  /**
   * Sets bias.
   *
   * @param bias the bias
   * @return the bias
   */
  public ImgBandBiasLayer setBias(Tensor bias) {
    if (this.bias != null) {
      this.bias.freeRef();
    }
    this.bias = bias;
    this.bias.addRef();
    return this;
  }
  
  @Override
  protected void _free() {
    if (this.bias != null) {
      bias.freeRef();
      bias = null;
    }
    super._free();
  }
}
