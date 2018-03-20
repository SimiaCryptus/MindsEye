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
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * This layer multiplies together the inputs, element-by-element. It can be used to implement integer-power activation
 * layers, such as the square needed in MeanSqLossLayer.
 */
@SuppressWarnings("serial")
public class SquareActivationLayer extends LayerBase implements MultiPrecision<SquareActivationLayer> {
  
  private Precision precision = Precision.Double;
  private double alpha = 1.0;
  
  /**
   * Instantiates a new Product inputs layer.
   */
  public SquareActivationLayer() {
  }
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param id the id
   */
  protected SquareActivationLayer(@Nonnull final JsonObject id) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
    this.alpha = id.getAsJsonPrimitive("alpha").getAsDouble();
  }
  
  /**
   * From json product inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the product inputs layer
   */
  public static SquareActivationLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new SquareActivationLayer(json);
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
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    if (inObj.length != 1) {
      throw new IllegalArgumentException("inObj.length=" + inObj.length);
    }
    Result input = inObj[0];
    final TensorList inputData = input.getData();
    @Nonnull final int[] dimensions = inputData.getDimensions();
    final int length = inputData.length();
    if (3 != dimensions.length) {
      throw new IllegalArgumentException("dimensions=" + Arrays.toString(dimensions));
    }
    return new Result(CudaSystem.run(gpu -> {
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
      @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
        dimensions[2], dimensions[1], dimensions[0],
        dimensions[2] * dimensions[1] * dimensions[0],
        dimensions[1] * dimensions[0],
        dimensions[0],
        1);
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
      //assert inputTensor.size == rPtr.size;
      @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length, MemoryType.Device, true);
      CudaMemory lPtrMemory = inputTensor.getMemory(gpu);
      CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(),
        precision.getPointer(alpha), inputTensor.descriptor.getPtr(), lPtrMemory.getPtr(),
        precision.getPointer(1.0), inputTensor.descriptor.getPtr(), lPtrMemory.getPtr(),
        precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      outputPtr.dirty();
      lPtrMemory.dirty();
      outputPtr.dirty();
      lPtrMemory.freeRef();
      inputTensor.freeRef();
      opDescriptor.freeRef();
      CudaTensor cudaTensor = CudaTensor.wrap(outputPtr, outputDescriptor, precision);
      return CudaTensorList.wrap(cudaTensor, length, dimensions, precision);
    }, inputData), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (input.isAlive()) {
        @Nonnull TensorList data = CudaSystem.run(gpu -> {
          @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
          @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
            dimensions[2], dimensions[1], dimensions[0],
            dimensions[2] * dimensions[1] * dimensions[0],
            dimensions[1] * dimensions[0],
            dimensions[0],
            1);
          @Nullable final CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, false);
          delta.freeRef();
          @Nullable final CudaTensor inputTensor = gpu.getTensor(input.getData(), precision, MemoryType.Device, false);
          //assert deltaTensor.size == inputTensor.size;
          @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length, MemoryType.Device, true);
          CudaMemory deltaTensorMemory = deltaTensor.getMemory(gpu);
          CudaMemory rightTensorMemory = inputTensor.getMemory(gpu);
          CudaSystem.handle(gpu.cudnnOpTensor(opDescriptor.getPtr(),
            precision.getPointer(2), deltaTensor.descriptor.getPtr(), deltaTensorMemory.getPtr(),
            precision.getPointer(alpha), inputTensor.descriptor.getPtr(), rightTensorMemory.getPtr(),
            precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
          deltaTensorMemory.dirty();
          rightTensorMemory.dirty();
          outputPtr.dirty();
          deltaTensorMemory.freeRef();
          rightTensorMemory.freeRef();
          CudaTensor cudaTensor = new CudaTensor(outputPtr, outputDescriptor, precision);
          Arrays.stream(new ReferenceCounting[]{deltaTensor, inputTensor, opDescriptor, outputDescriptor}).forEach(ReferenceCounting::freeRef);
          outputPtr.freeRef();
          return CudaTensorList.wrap(cudaTensor, length, dimensions, precision);
        }, delta);
        input.accumulate(buffer, data);
      }
      else {
        delta.freeRef();
      }
    }) {
  
      @Override
      public void accumulate(final DeltaSet<Layer> buffer, final TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }
  
  
      @Override
      protected void _free() {
        inputData.freeRef();
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
    json.addProperty("alpha", alpha);
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public SquareActivationLayer setPrecision(final Precision precision) {
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
  public SquareActivationLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }
}
