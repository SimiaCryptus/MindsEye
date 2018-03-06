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
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * This layer multiplies together the inputs, element-by-element. It can be used to implement integer-power activation
 * layers, such as the square needed in MeanSqLossLayer.
 */
@SuppressWarnings("serial")
public class NProductLayer extends LayerBase implements MultiPrecision<NProductLayer> {
  
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Product inputs layer.
   */
  public NProductLayer() {
  }
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param id the id
   */
  protected NProductLayer(@javax.annotation.Nonnull final JsonObject id) {
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
  public static NProductLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new NProductLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ProductInputsLayer.class);
  }
  
  
  @Nullable
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    if (inObj.length <= 1) {
      throw new IllegalArgumentException("inObj.length=" + inObj.length);
    }
    @Nonnull final int[] dimensions = inObj[0].getData().getDimensions();
    final int length = inObj[0].getData().length();
    if (3 != dimensions.length) {
      throw new IllegalArgumentException("dimensions=" + Arrays.toString(dimensions));
    }
    for (int i = 0; i < inObj.length; i++) {
      inObj[i].getData().addRef();
    }
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    for (int i = 1; i < inObj.length; i++) {
      TensorList data = inObj[i].getData();
      if (Tensor.length(dimensions) != Tensor.length(data.getDimensions())) {
        throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(data.getDimensions()));
      }
    }
    return new Result(CudaSystem.eval(gpu -> {
      @javax.annotation.Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
      @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision,
        length, dimensions[2], dimensions[1], dimensions[0],
        dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0], dimensions[0], 1);
      @javax.annotation.Nonnull final TensorList result1 = Arrays.stream(inObj).map(x -> {
        TensorList data = x.getData();
        data.addRef();
        return data;
      }).reduce((l, r) -> {
        @Nullable final CudaTensor lPtr = gpu.getTensor(l, precision, MemoryType.Device, false);
        @Nullable final CudaTensor rPtr = gpu.getTensor(r, precision, MemoryType.Device, false);
        //assert lPtr.memory.size == rPtr.memory.size;
        @javax.annotation.Nonnull final CudaMemory outputPtr = gpu.allocate((long) outputDescriptor.nStride * length * precision.size, MemoryType.Device, true);
        CudaMemory lPtrMemory = lPtr.getMemory(gpu);
        CudaMemory rPtrMemory = rPtr.getMemory(gpu);
        CudaSystem.handle(JCudnn.cudnnOpTensor(gpu.handle, opDescriptor.getPtr(),
          precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
          precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
          precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
        lPtrMemory.freeRef();
        rPtrMemory.freeRef();
        Arrays.stream(new ReferenceCounting[]{lPtr, rPtr, l, r}).forEach(ReferenceCounting::freeRef);
        outputDescriptor.addRef();
        return CudaTensorList.wrap(CudaTensor.wrap(outputPtr, outputDescriptor, precision), length, dimensions, precision);
      }).get();
      Arrays.stream(new ReferenceCounting[]{opDescriptor, outputDescriptor}).forEach(ReferenceCounting::freeRef);
      return result1;
    }, Arrays.stream(inObj).map(Result::getData).toArray()), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      for (int index = 0; index < inObj.length; index++) {
        final Result input = inObj[index];
        if (input.isAlive()) {
          final int _index = index;
          @javax.annotation.Nonnull TensorList data = IntStream.range(0, inObj.length).mapToObj(i -> {
            TensorList tensorList = i == _index ? delta : inObj[i].getData();
            tensorList.addRef();
            return tensorList;
          }).reduce((l, r) -> {
            return CudaSystem.eval(gpu -> {
              @javax.annotation.Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision);
              @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, dimensions[2], dimensions[1], dimensions[0], dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0], dimensions[0], 1);
  
              @Nullable final CudaTensor lPtr = gpu.getTensor(l, precision, MemoryType.Device, false);
              @Nullable final CudaTensor rPtr = gpu.getTensor(r, precision, MemoryType.Device, false);
              //assert lPtr.memory.size == rPtr.memory.size;
              @javax.annotation.Nonnull final CudaMemory outputPtr = gpu.allocate((long) outputDescriptor.nStride * length * precision.size, MemoryType.Device, true);
              CudaMemory lPtrMemory = lPtr.getMemory(gpu);
              CudaMemory rPtrMemory = rPtr.getMemory(gpu);
              CudaSystem.handle(JCudnn.cudnnOpTensor(gpu.handle, opDescriptor.getPtr(),
                precision.getPointer(1.0), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
                precision.getPointer(1.0), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
                precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr()));
              lPtrMemory.freeRef();
              rPtrMemory.freeRef();
              Arrays.stream(new ReferenceCounting[]{lPtr, rPtr, opDescriptor, l, r}).forEach(ReferenceCounting::freeRef);
              return CudaTensorList.wrap(CudaTensor.wrap(outputPtr, outputDescriptor, precision), length, dimensions, precision);
            }, l, r);
          }).get();
          input.accumulate(buffer, data);
        }
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
        for (int i = 0; i < inObj.length; i++) {
          inObj[i].getData().freeRef();
        }
      }
      
      
      @Override
      public boolean isAlive() {
        for (@javax.annotation.Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
      
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public NProductLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
