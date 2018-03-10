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
import com.simiacryptus.mindseye.lang.CoreSettings;
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
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;

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
  protected BinarySumLayer(@Nonnull final JsonObject json) {
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
  public static BinarySumLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new BinarySumLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    @Nonnull PipelineNetwork network = new PipelineNetwork(2);
    network.wrap(new SumInputsLayer(),
      network.wrap(new LinearActivationLayer().setScale(this.leftFactor).freeze(), network.getInput(0)),
      network.wrap(new LinearActivationLayer().setScale(this.rightFactor).freeze(), network.getInput(1)));
    return network;
    
  }
  
  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (inObj.length == 1) {
      if (rightFactor != 1) throw new IllegalStateException();
      if (leftFactor != 1) throw new IllegalStateException();
      return inObj[0];
    }
    if (inObj.length > 2) {
      if (rightFactor != 1) throw new IllegalStateException();
      if (leftFactor != 1) throw new IllegalStateException();
      return Arrays.stream(inObj).reduce((a, b) -> evalAndFree(a, b)).get();
    }
    assert (inObj.length == 2);
    final TensorList leftData = inObj[0].getData();
    final TensorList rightData = inObj[1].getData();
    @Nonnull final int[] dimensions = leftData.getDimensions();
    final int length = leftData.length();
    if (length != rightData.length()) throw new IllegalArgumentException();
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
      @Nonnull final CudaResource<cudnnOpTensorDescriptor> opDescriptor = gpu.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision);
      @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length,
        dimensions[2], dimensions[1], dimensions[0],
        dimensions[2] * dimensions[1] * dimensions[0],
        dimensions[1] * dimensions[0],
        dimensions[0],
        1);
      @Nullable final CudaTensor lPtr = gpu.getTensor(leftData, precision, MemoryType.Device, false); //.getDenseAndFree(gpu);//.moveTo(gpu.getDeviceNumber());
      @Nullable final CudaTensor rPtr = gpu.getTensor(rightData, precision, MemoryType.Device, false); //.getDenseAndFree(gpu);//.moveTo(gpu.getDeviceNumber());
      @Nonnull final CudaMemory outputPtr = gpu.allocate(precision.size * Tensor.length(dimensions) * length, MemoryType.Device, false);
      CudaMemory lPtrMemory = lPtr.getMemory(gpu);
      CudaMemory rPtrMemory = rPtr.getMemory(gpu);
      gpu.cudnnOpTensor(opDescriptor.getPtr(),
        precision.getPointer(leftFactor), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
        precision.getPointer(rightFactor), rPtr.descriptor.getPtr(), rPtrMemory.getPtr(),
        precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr());
      outputPtr.dirty(gpu);
      rPtrMemory.freeRef();
      lPtrMemory.freeRef();
      CudaTensor cudaTensor = CudaTensor.wrap(outputPtr, outputDescriptor, precision);
      Stream.<ReferenceCounting>of(opDescriptor, lPtr, rPtr).forEach(ReferenceCounting::freeRef);
      return CudaTensorList.wrap(cudaTensor, length, dimensions, precision);
    }, leftData), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      
      Runnable a = () -> {
        if (inObj[0].isAlive()) {
          CudaTensorList tensorList = CudaSystem.eval(gpu -> {
            @Nullable final CudaTensor lPtr = gpu.getTensor(delta, precision, MemoryType.Device, false);
            @Nonnull final CudaMemory passbackPtr = gpu.allocate(precision.size * Tensor.length(dimensions) * length, MemoryType.Managed, true);
            @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision, length,
              dimensions[2], dimensions[1], dimensions[0],
              dimensions[2] * dimensions[1] * dimensions[0],
              dimensions[1] * dimensions[0],
              dimensions[0],
              1);
            CudaMemory lPtrMemory = lPtr.getMemory(gpu);
            gpu.cudnnTransformTensor(
              precision.getPointer(leftFactor), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
              precision.getPointer(0.0), passbackDescriptor.getPtr(), passbackPtr.getPtr());
            passbackPtr.dirty(gpu);
            lPtrMemory.freeRef();
            CudaTensor cudaTensor = CudaTensor.wrap(passbackPtr, passbackDescriptor, precision);
            lPtr.freeRef();
            return CudaTensorList.wrap(cudaTensor, length, dimensions, precision);
          }, delta);
          inObj[0].accumulate(buffer, tensorList);
        }
      };
      Runnable b = () -> {
        if (inObj[1].isAlive()) {
          CudaTensorList tensorList = CudaSystem.eval(gpu -> {
            @Nullable final CudaTensor lPtr = gpu.getTensor(delta, precision, MemoryType.Device, false);
            @Nonnull final CudaMemory outputPtr = gpu.allocate(precision.size * Tensor.length(dimensions) * length, MemoryType.Managed, true);
            @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(precision, length,
              dimensions[2], dimensions[1], dimensions[0],
              dimensions[2] * dimensions[1] * dimensions[0],
              dimensions[1] * dimensions[0],
              dimensions[0],
              1);
            CudaMemory lPtrMemory = lPtr.getMemory(gpu);
            gpu.cudnnTransformTensor(
              precision.getPointer(rightFactor), lPtr.descriptor.getPtr(), lPtrMemory.getPtr(),
              precision.getPointer(0.0), passbackDescriptor.getPtr(), outputPtr.getPtr());
            lPtrMemory.freeRef();
            CudaTensor cudaTensor = CudaTensor.wrap(outputPtr, passbackDescriptor, precision);
            lPtr.freeRef();
            return CudaTensorList.wrap(cudaTensor, length, dimensions, precision);
          }, delta);
          inObj[1].accumulate(buffer, tensorList);
        }
      };
      if (CoreSettings.INSTANCE.isSingleThreaded()) TestUtil.runAllSerial(a, b);
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
    @Nonnull final JsonObject json = super.getJsonStub();
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
  @Nonnull
  public BinarySumLayer setLeftFactor(final double leftFactor) {
    this.leftFactor = leftFactor;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
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
  @Nonnull
  public BinarySumLayer setRightFactor(final double rightFactor) {
    this.rightFactor = rightFactor;
    return this;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
