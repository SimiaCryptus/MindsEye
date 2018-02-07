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
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Computes a weighted binary sum of two layers. Provides two weighting coefficients, one for each input. This can be
 * used to implement a summation layer, a difference layer, a scaling layer, or any combination.
 */
@SuppressWarnings("serial")
public class BinarySumLayer extends NNLayer implements MultiPrecision<BinarySumLayer> {
  
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
  protected BinarySumLayer(final @NotNull JsonObject json) {
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
  public static BinarySumLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new BinarySumLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public @NotNull NNLayer getCompatibilityLayer() {
    @NotNull PipelineNetwork network = new PipelineNetwork(2);
    network.add(new SumInputsLayer(),
                network.add(new LinearActivationLayer().setScale(this.leftFactor).freeze(), network.getInput(0)),
                network.add(new LinearActivationLayer().setScale(this.rightFactor).freeze(), network.getInput(1)));
    return network;
    
  }
  
  @Override
  public NNResult eval(final @NotNull NNResult... inObj) {
    if (inObj.length == 1) {
      if (rightFactor != 1) throw new IllegalStateException();
      if (leftFactor != 1) throw new IllegalStateException();
      return inObj[0];
    }
    if (inObj.length > 2) {
      if (rightFactor != 1) throw new IllegalStateException();
      if (leftFactor != 1) throw new IllegalStateException();
      for (@NotNull NNResult nnResult : inObj) {
        nnResult.addRef();
        nnResult.getData().addRef();
      }
      return Arrays.stream(inObj).reduce((a, b) -> {
        NNResult r = eval(a, b);
        a.freeRef();
        a.getData().freeRef();
        b.freeRef();
        b.getData().freeRef();
        return r;
      }).get();
    }
    assert (inObj.length == 2);
    final TensorList leftData = inObj[0].getData();
    final TensorList rightData = inObj[1].getData();
    final int[] dimensions = leftData.getDimensions();
    final int length = leftData.length();
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
    return new NNResult(GpuSystem.eval(gpu -> {
      final @NotNull CudaResource<cudnnOpTensorDescriptor> opDescriptor = GpuSystem.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision.code);
      final @NotNull CudaResource<cudnnTensorDescriptor> sizeDescriptor = GpuSystem.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
      final CudaPtr lPtr = CudaPtr.getCudaPtr(precision, leftData);//.moveTo(gpu.getDeviceNumber());
      final CudaPtr rPtr = CudaPtr.getCudaPtr(precision, rightData);//.moveTo(gpu.getDeviceNumber());
      assert lPtr.size == rPtr.size;
      final @NotNull CudaPtr outputPtr = CudaPtr.allocate(gpu.getDeviceNumber(), lPtr.size, MemoryType.Managed, true);
      CuDNNHandle.cudnnOpTensor(gpu.getHandle(), opDescriptor.getPtr(),
                                precision.getPointer(leftFactor), sizeDescriptor.getPtr(), lPtr.getPtr(),
                                precision.getPointer(rightFactor), sizeDescriptor.getPtr(), rPtr.getPtr(),
                                precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
      gpu.registerForCleanup(opDescriptor, sizeDescriptor, lPtr, rPtr);
      return GpuTensorList.wrap(outputPtr, length, dimensions, precision);
    }), (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList delta) -> {
      TestUtil.runAllSerial(() -> {
        if (inObj[0].isAlive()) {
          GpuTensorList tensorList = GpuSystem.eval(gpu -> {
            final CudaPtr lPtr = CudaPtr.getCudaPtr(precision, delta);
            final @NotNull CudaPtr outputPtr = CudaPtr.allocate(gpu.getDeviceNumber(), lPtr.size, MemoryType.Managed, true);
            final @NotNull CudaResource<cudnnTensorDescriptor> sizeDescriptor = GpuSystem.newTensorDescriptor(
              precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
            CuDNNHandle.cudnnAddTensor(gpu.getHandle(),
                                       precision.getPointer(leftFactor), sizeDescriptor.getPtr(), lPtr.getPtr(),
                                       precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
            gpu.registerForCleanup(sizeDescriptor, lPtr);
            return GpuTensorList.wrap(outputPtr, length, dimensions, precision);
          });
          inObj[0].accumulate(buffer, tensorList);
          tensorList.freeRef();
        }
      }, () -> {
        if (inObj[1].isAlive()) {
          GpuTensorList tensorList = GpuSystem.eval(gpu -> {
            final CudaPtr lPtr = CudaPtr.getCudaPtr(precision, delta);
            final @NotNull CudaPtr outputPtr = CudaPtr.allocate(gpu.getDeviceNumber(), lPtr.size, MemoryType.Managed, true);
            final @NotNull CudaResource<cudnnTensorDescriptor> sizeDescriptor = GpuSystem.newTensorDescriptor(
              precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
            CuDNNHandle.cudnnAddTensor(gpu.getHandle(),
                                       precision.getPointer(rightFactor), sizeDescriptor.getPtr(), lPtr.getPtr(),
                                       precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr());
            gpu.registerForCleanup(sizeDescriptor, lPtr);
            return GpuTensorList.wrap(outputPtr, length, dimensions, precision);
          });
          inObj[1].accumulate(buffer, tensorList);
          tensorList.freeRef();
        }
      });
    }) {
  
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(x -> x.freeRef());
      }
  
  
      @Override
      public boolean isAlive() {
        for (final @NotNull NNResult element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
  
    };
  }
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final @NotNull JsonObject json = super.getJsonStub();
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
  public @NotNull BinarySumLayer setLeftFactor(final double leftFactor) {
    this.leftFactor = leftFactor;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public @NotNull BinarySumLayer setPrecision(final Precision precision) {
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
  public @NotNull BinarySumLayer setRightFactor(final double rightFactor) {
    this.rightFactor = rightFactor;
    return this;
  }
  
  @Override
  public @NotNull List<double[]> state() {
    return Arrays.asList();
  }
}
