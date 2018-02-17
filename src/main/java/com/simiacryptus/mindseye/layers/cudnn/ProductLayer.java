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
import java.util.stream.IntStream;

/**
 * This layer multiplies together the inputs, element-by-element. It can be used to implement integer-power activation
 * layers, such as the square needed in MeanSqLossLayer.
 */
@SuppressWarnings("serial")
public class ProductLayer extends LayerBase implements MultiPrecision<ProductLayer> {
  
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Product inputs layer.
   */
  public ProductLayer() {
  }
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param id the id
   */
  protected ProductLayer(@javax.annotation.Nonnull final JsonObject id) {
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
  public static ProductLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ProductLayer(json);
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
    if (!GpuSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
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
      if (Tensor.dim(dimensions) != Tensor.dim(data.getDimensions())) {
        throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(data.getDimensions()));
      }
    }
    return new Result(GpuSystem.eval(gpu -> {
      @javax.annotation.Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = GpuSystem.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision.code);
      @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = GpuSystem.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
      @javax.annotation.Nonnull final TensorList result1 = Arrays.stream(inObj).map(x -> {
        TensorList data = x.getData();
        data.addRef();
        return data;
      }).reduce((l, r) -> {
        @Nullable final CudaPtr lPtr = CudaPtr.getCudaPtr(precision, l);
        @Nullable final CudaPtr rPtr = CudaPtr.getCudaPtr(precision, r);
        assert lPtr.size == rPtr.size;
        @javax.annotation.Nonnull final CudaPtr outputPtr = CudaPtr.allocate(gpu.getDeviceNumber(), lPtr.size, MemoryType.Managed, true);
        GpuSystem.handle(JCudnn.cudnnOpTensor(gpu.getHandle(), opDescriptor.getPtr(),
          precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr(),
          precision.getPointer(1.0), sizeDescriptor.getPtr(), rPtr.getPtr(),
          precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr()));
        gpu.registerForCleanup(lPtr, rPtr, l, r);
        return GpuTensorList.wrap(outputPtr, length, dimensions, precision);
      }).get();
      gpu.registerForCleanup(opDescriptor, sizeDescriptor);
      return result1;
    }), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      for (int index = 0; index < inObj.length; index++) {
        final Result input = inObj[index];
        if (input.isAlive()) {
          final int _index = index;
          @javax.annotation.Nonnull TensorList data = IntStream.range(0, inObj.length).mapToObj(i -> {
            TensorList tensorList = i == _index ? delta : inObj[i].getData();
            tensorList.addRef();
            return tensorList;
          }).reduce((l, r) -> {
            return GpuSystem.eval(gpu -> {
              @javax.annotation.Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = GpuSystem.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision.code);
              @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = GpuSystem.newTensorDescriptor(
                precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
              
              @Nullable final CudaPtr lPtr = CudaPtr.getCudaPtr(precision, l);
              @Nullable final CudaPtr rPtr = CudaPtr.getCudaPtr(precision, r);
              assert lPtr.size == rPtr.size;
              @javax.annotation.Nonnull final CudaPtr outputPtr = CudaPtr.allocate(gpu.getDeviceNumber(), lPtr.size, MemoryType.Managed, true);
              GpuSystem.handle(JCudnn.cudnnOpTensor(gpu.getHandle(), opDescriptor.getPtr(),
                precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr(),
                precision.getPointer(1.0), sizeDescriptor.getPtr(), rPtr.getPtr(),
                precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr()));
              gpu.registerForCleanup(lPtr, rPtr, opDescriptor, sizeDescriptor, l, r);
              return GpuTensorList.wrap(outputPtr, length, dimensions, precision);
            });
          }).get();
          input.accumulate(buffer, data);
          data.freeRef();
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
  public ProductLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
