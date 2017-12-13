/*
 * Copyright (c) 2017 by Andrew Charneski.
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
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnOpTensorOp;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;

import static jcuda.jcudnn.JCudnn.cudnnAddTensor;
import static jcuda.jcudnn.JCudnn.cudnnOpTensor;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Product inputs layer.
 */
public class BinarySumLayer extends NNLayer implements LayerPrecision<BinarySumLayer> {
  
  private Precision precision = Precision.Double;
  private double leftFactor = 1.0;
  private double rightFactor = 1.0;
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param json the id
   */
  protected BinarySumLayer(JsonObject json) {
    super(json);
    rightFactor = json.get("rightFactor").getAsDouble();
    leftFactor = json.get("leftFactor").getAsDouble();
  }
  
  /**
   * Instantiates a new Product inputs layer.
   */
  public BinarySumLayer() {
  }
  
  /**
   * Instantiates a new Binary sum layer.
   *
   * @param leftFactor  the left factor
   * @param rightFactor the right factor
   */
  public BinarySumLayer(double leftFactor, double rightFactor) {
    this.leftFactor = leftFactor;
    this.rightFactor = rightFactor;
  }
  
  /**
   * From json product inputs layer.
   *
   * @param json the json
   * @return the product inputs layer
   */
  public static BinarySumLayer fromJson(JsonObject json) {
    return new BinarySumLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("rightFactor", rightFactor);
    json.addProperty("leftFactor", leftFactor);
    return json;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    ((CudaExecutionContext) nncontext).initThread();
    if (inObj.length != 2) {
      throw new IllegalArgumentException("inObj.length=" + inObj.length);
    }
    NNResult l = inObj[0];
    TensorList leftData = l.getData();
    TensorList rightData = inObj[1].getData();
    int[] dimensions = leftData.getDimensions();
    int length = leftData.length();
    if (3 != dimensions.length) {
      throw new IllegalArgumentException("dimensions=" + Arrays.toString(dimensions));
    }
    for (int i = 1; i < inObj.length; i++) {
      if (Tensor.dim(dimensions) != Tensor.dim(inObj[i].getData().getDimensions())) {
        throw new IllegalArgumentException(Arrays.toString(dimensions) + " != " + Arrays.toString(inObj[i].getData().getDimensions()));
      }
    }
    
    final CudaResource<cudnnOpTensorDescriptor> opDescriptor = CuDNN.newOpDescriptor(cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD, precision.code);
    CudaResource<cudnnTensorDescriptor> sizeDescriptor = CuDNN.newTensorDescriptor(
      precision.code, CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
    CudaPtr lPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, leftData);
    CudaPtr rPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, rightData);
    assert lPtr.size == rPtr.size;
    CudaPtr outputPtr = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), lPtr.size);
    CuDNN.handle(cudnnOpTensor(((CuDNN) nncontext).cudnnHandle, opDescriptor.getPtr(),
      precision.getPointer(leftFactor), sizeDescriptor.getPtr(), lPtr.getPtr(),
      precision.getPointer(rightFactor), sizeDescriptor.getPtr(), rPtr.getPtr(),
      precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr()));
    TensorList result = new GpuTensorList(outputPtr, length, dimensions, ((CuDNN) nncontext).cudnnHandle, this.precision);
    
    return new NNResult(result) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList delta) {
        ((CudaExecutionContext) nncontext).initThread();
        assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        
        if (inObj[0].isAlive()) {
          CudaPtr lPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), BinarySumLayer.this.precision, delta);
          CudaPtr outputPtr = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), lPtr.size);
          CuDNN.handle(cudnnAddTensor(((CuDNN) nncontext).cudnnHandle,
            precision.getPointer(leftFactor), sizeDescriptor.getPtr(), lPtr.getPtr(),
            precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr()));
          TensorList data = new GpuTensorList(outputPtr, length, dimensions, ((CuDNN) nncontext).cudnnHandle, BinarySumLayer.this.precision);
          inObj[0].accumulate(buffer, data);
        }
        
        if (inObj[1].isAlive()) {
          CudaPtr lPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), BinarySumLayer.this.precision, delta);
          CudaPtr outputPtr = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), lPtr.size);
          CuDNN.handle(cudnnAddTensor(((CuDNN) nncontext).cudnnHandle,
            precision.getPointer(rightFactor), sizeDescriptor.getPtr(), lPtr.getPtr(),
            precision.getPointer(0.0), sizeDescriptor.getPtr(), outputPtr.getPtr()));
          TensorList data = new GpuTensorList(outputPtr, length, dimensions, ((CuDNN) nncontext).cudnnHandle, BinarySumLayer.this.precision);
          inObj[1].accumulate(buffer, data);
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
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  public Precision getPrecision() {
    return precision;
  }
  
  public BinarySumLayer setPrecision(Precision precision) {
    this.precision = precision;
    return this;
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
  public BinarySumLayer setLeftFactor(double leftFactor) {
    this.leftFactor = leftFactor;
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
  public BinarySumLayer setRightFactor(double rightFactor) {
    this.rightFactor = rightFactor;
    return this;
  }
}
