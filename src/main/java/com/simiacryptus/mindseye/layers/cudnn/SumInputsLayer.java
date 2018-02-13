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
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcudnn.cudnnTensorFormat;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Computes a weighted binary sum of two layers. Provides two weighting coefficients, one for each input. This can be
 * used to implement a summation layer, a difference layer, a scaling layer, or any combination.
 */
@SuppressWarnings("serial")
public class SumInputsLayer extends NNLayer implements MultiPrecision<SumInputsLayer> {
  
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Product inputs layer.
   */
  public SumInputsLayer() {
    super();
  }
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param json the id
   */
  protected SumInputsLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * From json product inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the product inputs layer
   */
  public static SumInputsLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new SumInputsLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public NNLayer getCompatibilityLayer() {
    @javax.annotation.Nonnull PipelineNetwork network = new PipelineNetwork(2);
    network.wrap(new com.simiacryptus.mindseye.layers.java.SumInputsLayer(),
      network.wrap(new LinearActivationLayer().setScale(1.0).freeze(), network.getInput(0)),
      network.wrap(new LinearActivationLayer().setScale(1.0).freeze(), network.getInput(1)));
    return network;
    
  }
  
  @Nullable
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    @Nonnull final int[] dimensions = inObj[0].getData().getDimensions();
    final int length = inObj[0].getData().length();
    if (3 != dimensions.length) {
      throw new IllegalArgumentException("dimensions=" + Arrays.toString(dimensions));
    }
    for (int i = 1; i < inObj.length; i++) {
      if (Tensor.dim(dimensions) != Tensor.dim(inObj[i].getData().getDimensions())) {
        throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(inObj[i].getData().getDimensions()));
      }
    }
    if (!GpuSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    Arrays.stream(inObj).forEach(x -> x.addRef());
    //stream0 = stream0.parallel();
    @javax.annotation.Nonnull TensorList run = Arrays.stream(inObj).map(x -> x.getData()).map(x -> {
      x.addRef();
      return x;
    }).reduce((leftData, rightData) -> GpuSystem.eval(gpu -> {
      @javax.annotation.Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = GpuSystem.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision.code);
      @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = GpuSystem.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
      @Nullable final CudaPtr lPtr = CudaPtr.getCudaPtr(precision, leftData);//.moveTo(gpu.getDeviceNumber());
      @Nullable final CudaPtr rPtr = CudaPtr.getCudaPtr(precision, rightData);//.moveTo(gpu.getDeviceNumber());
      assert lPtr.size == rPtr.size;
      @javax.annotation.Nonnull final CudaPtr outputPtr = CudaPtr.allocate(gpu.getDeviceNumber(), lPtr.size, MemoryType.Managed, true);
      gpu.cudnnOpTensor(opDescriptor.getPtr(),
        precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr(),
        precision.getPointer(1.0), sizeDescriptor.getPtr(), rPtr.getPtr(),
        precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
      gpu.registerForCleanup(lPtr, rPtr, opDescriptor, sizeDescriptor, leftData, rightData);
      return GpuTensorList.wrap(outputPtr, length, dimensions, precision);
    })).get();
    return new NNResult(run, (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      @javax.annotation.Nonnull Stream<NNResult> stream1 = Arrays.stream(inObj);
      // TODO: Fix issue where parallel will cause data corruption
      if (!TestUtil.CONSERVATIVE) stream1 = stream1.parallel();
      stream1.filter(x -> x.isAlive()).forEach(obj -> {
        GpuTensorList tensorList = GpuSystem.eval(gpu -> {
          @Nullable final CudaPtr lPtr = CudaPtr.getCudaPtr(precision, delta);
          @javax.annotation.Nonnull final CudaPtr outputPtr = CudaPtr.allocate(gpu.getDeviceNumber(), lPtr.size, MemoryType.Managed, true);
          @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = GpuSystem.newTensorDescriptor(
            precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
          gpu.cudnnAddTensor(
            precision.getPointer(1.0), sizeDescriptor.getPtr(), lPtr.getPtr(),
            precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
          gpu.registerForCleanup(lPtr, sizeDescriptor);
          return GpuTensorList.wrap(outputPtr, length, dimensions, precision);
        });
        obj.accumulate(buffer, tensorList);
        tensorList.freeRef();
      });
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(x -> x.freeRef());
      }
      
      
      @Override
      public boolean isAlive() {
        for (@javax.annotation.Nonnull final NNResult element : inObj)
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
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public SumInputsLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
