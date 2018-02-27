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
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
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

/**
 * Computes a weighted binary sum of two layers. Provides two weighting coefficients, one for each input. This can be
 * used to implement a summation layer, a difference layer, a scaling layer, or any combination.
 */
@SuppressWarnings("serial")
public class BinarySumLayer extends LayerBase implements MultiPrecision<BinarySumLayer> {
  
  private double leftFactor;
  private Precision precision = Precision.Double;
  private double rightFactor;
  
  /**
   * Instantiates a new Product inputs layer.
   */
  public BinarySumLayer() {
    this(1.0, 1.0);
  }
  
  /**
   * Instantiates a new Binary sum layer.
   *
   * @param leftFactor  the left factor
   * @param rightFactor the right factor
   */
  public BinarySumLayer(final double leftFactor, final double rightFactor) {
    this.leftFactor = leftFactor;
    this.rightFactor = rightFactor;
  }
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param json the id
   */
  protected BinarySumLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    rightFactor = json.get("rightFactor").getAsDouble();
    leftFactor = json.get("leftFactor").getAsDouble();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * From json product inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the product inputs layer
   */
  public static BinarySumLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new BinarySumLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public Layer getCompatibilityLayer() {
    @javax.annotation.Nonnull PipelineNetwork network = new PipelineNetwork(2);
    network.wrap(new SumInputsLayer(),
      network.wrap(new LinearActivationLayer().setScale(this.leftFactor).freeze(), network.getInput(0)),
      network.wrap(new LinearActivationLayer().setScale(this.rightFactor).freeze(), network.getInput(1)));
    return network;
    
  }
  
  @Nullable
  @Override
  public Result evalAndFree(@javax.annotation.Nonnull final Result... inObj) {
    if (inObj.length == 1) {
      if (rightFactor != 1) throw new IllegalStateException();
      if (leftFactor != 1) throw new IllegalStateException();
      return inObj[0];
    }
    if (inObj.length > 2) {
      if (rightFactor != 1) throw new IllegalStateException();
      if (leftFactor != 1) throw new IllegalStateException();
      return Arrays.stream(inObj).reduce((a, b) -> {
        @Nullable Result r = evalAndFree(a, b);
        return r;
      }).get();
    }
    assert (inObj.length == 2);
    final TensorList leftData = inObj[0].getData();
    final TensorList rightData = inObj[1].getData();
    @Nonnull final int[] dimensions = leftData.getDimensions();
    final int length = leftData.length();
    if (3 != dimensions.length) {
      throw new IllegalArgumentException("dimensions=" + Arrays.toString(dimensions));
    }
    for (int i = 1; i < inObj.length; i++) {
      if (Tensor.length(dimensions) != Tensor.length(inObj[i].getData().getDimensions())) {
        throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(inObj[i].getData().getDimensions()));
      }
    }
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    return new Result(CudaSystem.eval(gpu -> {
      @javax.annotation.Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision.code);
      @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = gpu.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
      @Nullable final CudaTensor lPtr = gpu.getTensor(leftData, precision, MemoryType.Device);//.moveTo(gpu.getDeviceNumber());
      @Nullable final CudaTensor rPtr = gpu.getTensor(rightData, precision, MemoryType.Device);//.moveTo(gpu.getDeviceNumber());
      assert lPtr.memory.size == rPtr.memory.size;
      @javax.annotation.Nonnull final CudaMemory outputPtr = gpu.allocate(lPtr.memory.size, MemoryType.Device, true);
      gpu.cudnnOpTensor(opDescriptor.getPtr(),
        precision.getPointer(leftFactor), sizeDescriptor.getPtr(), lPtr.memory.getPtr(),
        precision.getPointer(rightFactor), sizeDescriptor.getPtr(), rPtr.memory.getPtr(),
        precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
      CudaTensor cudaTensor = new CudaTensor(outputPtr, sizeDescriptor);
      outputPtr.freeRef();
      Arrays.stream(new ReferenceCounting[]{opDescriptor, sizeDescriptor, lPtr, rPtr}).forEach(ReferenceCounting::freeRef);
      return CudaTensorList.wrap(cudaTensor, length, dimensions, precision);
    }), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
  
      Runnable a = () -> {
        if (inObj[0].isAlive()) {
          CudaTensorList tensorList = CudaSystem.eval(gpu -> {
            @Nullable final CudaTensor lPtr = gpu.getTensor(delta, precision, MemoryType.Device);
            @Nonnull final CudaMemory outputPtr = gpu.allocate(lPtr.memory.size, MemoryType.Managed, true);
            @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = gpu.newTensorDescriptor(
              precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
            gpu.cudnnAddTensor(
              precision.getPointer(leftFactor), sizeDescriptor.getPtr(), lPtr.memory.getPtr(),
              precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
            CudaTensor cudaTensor = CudaTensor.wrap(outputPtr, sizeDescriptor);
            lPtr.freeRef();
            return CudaTensorList.wrap(cudaTensor, length, dimensions, precision);
          });
          inObj[0].accumulate(buffer, tensorList);
        }
      };
      Runnable b = () -> {
        if (inObj[1].isAlive()) {
          CudaTensorList tensorList = CudaSystem.eval(gpu -> {
            @Nullable final CudaTensor lPtr = gpu.getTensor(delta, precision, MemoryType.Device);
            @Nonnull final CudaMemory outputPtr = gpu.allocate(lPtr.memory.size, MemoryType.Managed, true);
            @Nonnull final CudaResource<cudnnTensorDescriptor> sizeDescriptor = gpu.newTensorDescriptor(
              precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
            gpu.cudnnAddTensor(
              precision.getPointer(rightFactor), sizeDescriptor.getPtr(), lPtr.memory.getPtr(),
              precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
            CudaTensor cudaTensor = CudaTensor.wrap(outputPtr, sizeDescriptor);
            lPtr.freeRef();
            return CudaTensorList.wrap(cudaTensor, length, dimensions, precision);
          });
          inObj[1].accumulate(buffer, tensorList);
        }
      };
      if (CoreSettings.INSTANCE.isConservative()) TestUtil.runAllSerial(a, b);
      else TestUtil.runAllParallel(a, b);
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(x -> x.freeRef());
        leftData.freeRef();
        rightData.freeRef();
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
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("rightFactor", rightFactor);
    json.addProperty("leftFactor", leftFactor);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  /**
   * Gets left factor.
   *
   * @return the left factor
   */
  public double getLeftFactor() {
    return leftFactor;
  }
  
  /**
   * Sets left factor.
   *
   * @param leftFactor the left factor
   * @return the left factor
   */
  @javax.annotation.Nonnull
  public BinarySumLayer setLeftFactor(final double leftFactor) {
    this.leftFactor = leftFactor;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public BinarySumLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  /**
   * Gets right factor.
   *
   * @return the right factor
   */
  public double getRightFactor() {
    return rightFactor;
  }
  
  /**
   * Sets right factor.
   *
   * @param rightFactor the right factor
   * @return the right factor
   */
  @javax.annotation.Nonnull
  public BinarySumLayer setRightFactor(final double rightFactor) {
    this.rightFactor = rightFactor;
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
