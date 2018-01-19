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

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * This layer multiplies together the inputs, element-by-element. It can be used to implement integer-power activation
 * layers, such as the square needed in MeanSqLossLayer.
 */
@SuppressWarnings("serial")
public class ProductLayer extends NNLayer implements LayerPrecision<ProductLayer> {
  
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
  protected ProductLayer(final JsonObject id) {
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
  public static ProductLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ProductLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public NNLayer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ProductInputsLayer.class);
  }
  
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    if (!CuDNN.isEnabled()) return getCompatibilityLayer().eval(inObj);
    return CuDNN.run(nncontext -> {
      nncontext.initThread();
      if (inObj.length <= 1) {
        throw new IllegalArgumentException("inObj.length=" + inObj.length);
      }
      final int[] dimensions = inObj[0].getData().getDimensions();
      final int length = inObj[0].getData().length();
      if (3 != dimensions.length) {
        throw new IllegalArgumentException("dimensions=" + Arrays.toString(dimensions));
      }
      for (int i = 1; i < inObj.length; i++) {
        if (Tensor.dim(dimensions) != Tensor.dim(inObj[i].getData().getDimensions())) {
          throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(inObj[i].getData().getDimensions()));
        }
      }
      final CudaResource<cudnnOpTensorDescriptor> opDescriptor = CuDNN.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_MUL, precision.code);
      final CudaResource<cudnnTensorDescriptor> sizeDescriptor = CuDNN.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
      final TensorList result = Arrays.stream(inObj).map(x -> x.getData()).reduce((l, r) -> {
        final CudaPtr lPtr = CudaPtr.write(nncontext.getDeviceNumber(), precision, l);
        final CudaPtr rPtr = CudaPtr.write(nncontext.getDeviceNumber(), precision, r);
        assert lPtr.size == rPtr.size;
        final CudaPtr outputPtr = CuDNN.alloc(nncontext.getDeviceNumber(), lPtr.size, true);
        CuDNN.handle(JCudnn.cudnnOpTensor(nncontext.cudnnHandle, opDescriptor.getPtr(),
                                          precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr(),
                                          precision.getPointer(1.0), sizeDescriptor.getPtr(), rPtr.getPtr(),
                                          precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr()));
        return GpuTensorList.create(outputPtr, length, dimensions, precision);
      }).get();
    
      return new NNResult(result) {
      
        @Override
        public void free() {
          Arrays.stream(inObj).forEach(NNResult::free);
        }
      
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList delta) {
          assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
          for (int index = 0; index < inObj.length; index++) {
            final NNResult input = inObj[index];
            if (input.isAlive()) {
              final int _index = index;
              TensorList data = IntStream.range(0, inObj.length).mapToObj(i -> i == _index ? delta : inObj[i].getData()).reduce((l, r) -> {
                return CuDNN.run(nncontext -> {
                  nncontext.initThread();
                  final CudaPtr lPtr = CudaPtr.write(nncontext.getDeviceNumber(), precision, l);
                  final CudaPtr rPtr = CudaPtr.write(nncontext.getDeviceNumber(), precision, r);
                  assert lPtr.size == rPtr.size;
                  final CudaPtr outputPtr = CuDNN.alloc(nncontext.getDeviceNumber(), lPtr.size, true);
                  CuDNN.handle(JCudnn.cudnnOpTensor(nncontext.cudnnHandle, opDescriptor.getPtr(),
                                                    precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr(),
                                                    precision.getPointer(1.0), sizeDescriptor.getPtr(), rPtr.getPtr(),
                                                    precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr()));
                  return GpuTensorList.create(outputPtr, length, dimensions, precision);
                });
              }).get();
              input.accumulate(buffer, data);
            }
          }
        }
      
        @Override
        public boolean isAlive() {
          for (final NNResult element : inObj)
            if (element.isAlive()) {
              return true;
            }
          return false;
        }
      
      };
    });
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public ProductLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
