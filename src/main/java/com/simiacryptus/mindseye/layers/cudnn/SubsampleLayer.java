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
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;

import static com.simiacryptus.mindseye.layers.cudnn.CuDNN.cudnnTransformTensor;

/**
 * The type Img concat layer.
 */
public class SubsampleLayer extends NNLayer implements LayerPrecision<SubsampleLayer> {
  
  private int maxBands = -1;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   */
  protected SubsampleLayer(JsonObject json) {
    super(json);
    maxBands = json.get("maxBands").getAsInt();
  }
  
  /**
   * Instantiates a new Img concat layer.
   */
  public SubsampleLayer() {
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @return the img concat layer
   */
  public static SubsampleLayer fromJson(JsonObject json) {
    return new SubsampleLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("maxBands", maxBands);
    return json;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    assert 3 == inObj[0].getData().getDimensions().length;
    int[] dimOut = Arrays.copyOf(inObj[0].getData().getDimensions(), 3);
    int length = inObj[0].getData().length();
    assert Arrays.stream(inObj).allMatch(x -> 3 == x.getData().getDimensions().length && x.getData().getDimensions()[0] == dimOut[0] && x.getData().getDimensions()[1] == dimOut[1] && x.getData().length() == length);
    dimOut[2] = Arrays.stream(inObj).mapToInt(x -> x.getData().getDimensions()[2]).sum();
    if (0 < maxBands && dimOut[2] > maxBands) dimOut[2] = maxBands;
    ((CudaExecutionContext) nncontext).initThread();
    CudaPtr outputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), length * dimOut[2] * dimOut[1] * dimOut[0] * precision.size);
    int bandOffset = 0;
    for (int i = 0; i < inObj.length; i++) {
      TensorList data = inObj[i].getData();
      int[] dimensions = data.getDimensions();
      int bands = maxBands <= 0 ? dimensions[2] : Math.min(dimensions[2], maxBands - bandOffset);
      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, length, bands, dimensions[1], dimensions[0],
        dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0], dimensions[0], 1);
      CudaResource<cudnnTensorDescriptor> viewDescriptor = CuDNN.newTensorDescriptor(
        precision.code, length, bands, dimensions[1], dimensions[0],
        dimOut[2] * dimOut[1] * dimOut[0], dimOut[1] * dimOut[0], dimOut[0], 1);
      CudaPtr cudaPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, data);
      cudnnTransformTensor(((CuDNN) nncontext).cudnnHandle,
        precision.getPointer(1.0), inputDescriptor.getPtr(), cudaPtr.getPtr(),
        precision.getPointer(0.0), viewDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(dimensions[1] * dimensions[0] * bandOffset * precision.size)
      );
      bandOffset += bands;
    }
    TensorList outputData = new GpuTensorList(outputBuffer, length, dimOut, ((CuDNN) nncontext).cudnnHandle, precision);
    //assert outputData.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    return new NNResult(outputData) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList error) {
        if (!Arrays.equals(error.getDimensions(), outputData.getDimensions())) {
          throw new AssertionError(Arrays.toString(error.getDimensions()) + " != " + Arrays.toString(outputData.getDimensions()));
        }
        //outputBuffer.finalize();
        ((CudaExecutionContext) nncontext).initThread();
        assert (error.length() == inObj[0].getData().length());
        //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
        CudaPtr errorPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, error);
        int bandOffset = 0;
        for (int i = 0; i < inObj.length; i++) {
          NNResult input = inObj[i];
          int[] dimensions = input.getData().getDimensions();
          int bands = maxBands <= 0 ? dimensions[2] : Math.min(dimensions[2], maxBands - bandOffset);
          if (input.isAlive()) {
            int _bandOffset = bandOffset;
            CudaPtr passbackBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), length * dimensions[2] * dimensions[1] * dimensions[0] * precision.size);
            CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
              precision.code, length, bands, dimensions[1], dimensions[0],
              dimensions[2] * dimensions[1] * dimensions[0], dimensions[1] * dimensions[0], dimensions[0], 1);
            CudaResource<cudnnTensorDescriptor> viewDescriptor = CuDNN.newTensorDescriptor(
              precision.code, length, bands, dimensions[1], dimensions[0],
              dimOut[2] * dimOut[1] * dimOut[0], dimOut[1] * dimOut[0], dimOut[0], 1);
            cudnnTransformTensor(((CuDNN) nncontext).cudnnHandle,
              precision.getPointer(1.0), viewDescriptor.getPtr(), errorPtr.getPtr().withByteOffset(dimensions[1] * dimensions[0] * _bandOffset * precision.size),
              precision.getPointer(0.0), inputDescriptor.getPtr(), passbackBuffer.getPtr()
            );
            TensorList passbackTensorList = new GpuTensorList(passbackBuffer, length, dimensions, ((CuDNN) nncontext).cudnnHandle, precision);
            //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
            input.accumulate(buffer, passbackTensorList);
            passbackBuffer.finalize();
          }
          bandOffset += bands;
        }
      }
      
      @Override
      public boolean isAlive() {
        return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
      }
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  /**
   * Gets max bands.
   *
   * @return the max bands
   */
  public int getMaxBands() {
    return maxBands;
  }
  
  /**
   * Sets max bands.
   *
   * @param maxBands the max bands
   * @return the max bands
   */
  public SubsampleLayer setMaxBands(int maxBands) {
    this.maxBands = maxBands;
    return this;
  }
  
  public Precision getPrecision() {
    return precision;
  }
  
  public SubsampleLayer setPrecision(Precision precision) {
    this.precision = precision;
    return this;
  }
}
