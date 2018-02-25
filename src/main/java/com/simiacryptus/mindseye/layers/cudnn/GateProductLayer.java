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
import jcuda.jcudnn.*;

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
public class GateProductLayer extends LayerBase implements MultiPrecision<GateProductLayer> {
  
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Product inputs layer.
   */
  public GateProductLayer() {
  }
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param id the id
   */
  protected GateProductLayer(@Nonnull final JsonObject id) {
    super(id);
    this.precision = Precision.valueOf(id.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json product inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the product inputs layer
   */
  public static GateProductLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new GateProductLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ProductInputsLayer.class);
  }
  
  
  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    if (inObj.length != 2) {
      throw new IllegalArgumentException("inObj.length=" + inObj.length);
    }
    Result left = inObj[0];
    Result right = inObj[1];
    final TensorList leftData = left.getData();
    final TensorList rightData = right.getData();
    @Nonnull final int[] leftDimensions = leftData.getDimensions();
    @Nonnull final int[] rightDimensions = rightData.getDimensions();
    final int length = leftData.length();
    if (3 != leftDimensions.length) {
      throw new IllegalArgumentException("dimensions=" + Arrays.toString(leftDimensions));
    }
    leftData.addRef();
    rightData.addRef();
    left.addRef();
    right.addRef();
//   assert !right.isAlive();
    return new Result(CudaSystem.eval(gpu -> {
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision.code);
      @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptorL = gpu.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, leftDimensions[2], leftDimensions[1], leftDimensions[0]);
      @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptorR = gpu.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, rightDimensions[2], rightDimensions[1], rightDimensions[0]);
      @Nullable final CudaMemory lPtr = gpu.getPtr(leftData, precision, MemoryType.Device);
      @Nullable final CudaMemory rPtr = gpu.getPtr(rightData, precision, MemoryType.Device);
      //assert lPtr.size == rPtr.size;
      @Nonnull final CudaMemory outputPtr = gpu.allocate(lPtr.size, MemoryType.Device, true);
      CudaSystem.handle(JCudnn.cudnnOpTensor(gpu.getHandle(), opDescriptor.getPtr(),
        precision.getPointer(1.0), sizeDescriptorL.getPtr(), lPtr.getPtr(),
        precision.getPointer(1.0), sizeDescriptorR.getPtr(), rPtr.getPtr(),
        precision.getPointer(0.0), sizeDescriptorL.getPtr(), outputPtr.getPtr()));
      rPtr.freeRef();
      lPtr.freeRef();
      Arrays.stream(new ReferenceCounting[]{opDescriptor, sizeDescriptorL}).forEach(ReferenceCounting::freeRef);
      return CudaTensorList.wrap(outputPtr, length, leftDimensions, precision);
    }), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      for (int index = 0; index < inObj.length; index++) {
        if (left.isAlive()) {
          @Nonnull TensorList data = CudaSystem.eval(gpu -> {
            @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision.code);
            @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptorL = gpu.newTensorDescriptor(
              precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, leftDimensions[2], leftDimensions[1], leftDimensions[0]);
            @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptorR = gpu.newTensorDescriptor(
              precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, rightDimensions[2], rightDimensions[1], rightDimensions[0]);
            
            @Nullable final CudaMemory deltaPtr = gpu.getPtr(delta, precision, MemoryType.Device);
            @Nullable final CudaMemory rPtr = gpu.getPtr(right.getData(), precision, MemoryType.Device);
            //assert deltaPtr.size == rPtr.size;
            @Nonnull final CudaMemory outputPtr = gpu.allocate(deltaPtr.size, MemoryType.Device, true);
            CudaSystem.handle(JCudnn.cudnnOpTensor(gpu.getHandle(), opDescriptor.getPtr(),
              precision.getPointer(1.0), sizeDescriptorL.getPtr(), deltaPtr.getPtr(),
              precision.getPointer(1.0), sizeDescriptorR.getPtr(), rPtr.getPtr(),
              precision.getPointer(0.0), sizeDescriptorL.getPtr(), outputPtr.getPtr()));
            Arrays.stream(new ReferenceCounting[]{deltaPtr, rPtr, opDescriptor, sizeDescriptorL, sizeDescriptorR}).forEach(ReferenceCounting::freeRef);
            return CudaTensorList.wrap(outputPtr, length, leftDimensions, precision);
          });
          left.accumulate(buffer, data);
          data.freeRef();
        }
      }
    }) {
      
      @Override
      protected void _free() {
        leftData.freeRef();
        rightData.freeRef();
        left.freeRef();
        right.freeRef();
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
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public GateProductLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
