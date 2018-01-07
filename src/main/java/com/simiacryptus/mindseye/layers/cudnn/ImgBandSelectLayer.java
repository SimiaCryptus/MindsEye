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
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Concatenates two or more inputs, assuming they have the same width and height, to produce an image with both inputs'
 * color bands. (e.g. Used in Inception modules in GoogLeNet.)
 */
@SuppressWarnings("serial")
public class ImgBandSelectLayer extends NNLayer implements LayerPrecision<ImgBandSelectLayer> {
  
  private int from;
  private int to;
  private Precision precision = Precision.Double;
  
  public ImgBandSelectLayer(int from, int to) {
    this.setFrom(from);
    this.setTo(to);
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   */
  protected ImgBandSelectLayer(final JsonObject json) {
    super(json);
    setFrom(json.get("from").getAsInt());
    setTo(json.get("to").getAsInt());
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static ImgBandSelectLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ImgBandSelectLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public NNLayer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer(IntStream.range(getFrom(), getTo()).toArray());
  }
  
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    assert getFrom() < getTo();
    assert getFrom() > 0;
    assert getTo() > 0;
    assert 1 == inObj.length;
    assert 3 == inObj[0].getData().getDimensions().length;
    if (((CudaExecutionContext) nncontext).getDeviceNumber() < 0) return getCompatibilityLayer().eval(nncontext, inObj);
    final TensorList inputData = inObj[0].getData();
    final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    final int[] outputDimensions = Arrays.copyOf(inputDimensions, 3);
    outputDimensions[2] = getTo() - getFrom();
    
    ((CudaExecutionContext) nncontext).initThread();
    final CudaPtr cudaOutput = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size, true);
    final CudaPtr cudaInput = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, inputData);
    final CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
      precision.code, length, outputDimensions[2], outputDimensions[1], outputDimensions[0], //
      inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
      inputDimensions[1] * inputDimensions[0], //
      inputDimensions[0], //
      1);
    final CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
      precision.code, length, outputDimensions[2], outputDimensions[1], outputDimensions[0], //
      outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
      outputDimensions[1] * outputDimensions[0], //
      outputDimensions[0], //
      1);
    final int byteOffset = inputDimensions[1] * inputDimensions[0] * getFrom() * precision.size;
    CuDNN.cudnnTransformTensor(((CuDNN) nncontext).cudnnHandle,
                               precision.getPointer(1.0), inputDescriptor.getPtr(), cudaInput.getPtr().withByteOffset(byteOffset),
                               precision.getPointer(0.0), outputDescriptor.getPtr(), cudaOutput.getPtr()
                              );
    final TensorList outputData = new GpuTensorList(cudaOutput, length, outputDimensions, ((CuDNN) nncontext).cudnnHandle, precision);
    //assert outputData.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    ManagedCudaPtr cleanupPtr = cudaOutput.managed(PersistanceMode.Weak);
    return new NNResult(outputData) {
      
      @Override
      public void free() {
        Arrays.stream(inObj).forEach(NNResult::free);
        cleanupPtr.free();
      }
      
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
        if (!Arrays.equals(error.getDimensions(), outputData.getDimensions())) {
          throw new AssertionError(Arrays.toString(error.getDimensions()) + " != " + Arrays.toString(outputData.getDimensions()));
        }
        if (inObj[0].isAlive()) {
          ((CudaExecutionContext) nncontext).initThread();
          assert error.length() == inputData.length();
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
          final CudaPtr errorPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, error);
          final CudaPtr passbackBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), //
                                                     (long) (length * inputDimensions[2] * inputDimensions[1] * inputDimensions[0] * precision.size), //
                                                     CudaPtr.MemoryType.DeviceDirect, true);
          CuDNN.cudnnTransformTensor(((CuDNN) nncontext).cudnnHandle,
                                     precision.getPointer(1.0), outputDescriptor.getPtr(), errorPtr.getPtr(),
                                     precision.getPointer(0.0), inputDescriptor.getPtr(), passbackBuffer.getPtr().withByteOffset(byteOffset)
                                    );
          final TensorList passbackTensorList = new GpuTensorList(passbackBuffer, length, inputDimensions, ((CuDNN) nncontext).cudnnHandle, precision);
          //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          inObj[0].accumulate(buffer, passbackTensorList);
          passbackBuffer.finalize();
        }
      }
      
      @Override
      public boolean isAlive() {
        return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
      }
    };
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.addProperty("from", getFrom());
    json.addProperty("to", getTo());
    json.addProperty("precision", precision.name());
    return json;
  }
  
  /**
   * Gets max bands.
   *
   * @return the max bands
   */
  public int getFrom() {
    return from;
  }
  
  /**
   * Sets max bands.
   *
   * @param from the max bands
   * @return the max bands
   */
  public ImgBandSelectLayer setFrom(final int from) {
    this.from = from;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public ImgBandSelectLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  public int getTo() {
    return to;
  }
  
  public ImgBandSelectLayer setTo(int to) {
    this.to = to;
    return this;
  }
}
