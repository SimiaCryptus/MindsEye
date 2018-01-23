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
  
  /**
   * Instantiates a new Img band select layer.
   *
   * @param from the from
   * @param to   the to
   */
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
  public NNResult eval(final NNResult... inObj) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    assert getFrom() < getTo();
    assert getFrom() >= 0;
    assert getTo() > 0;
    assert 1 == inObj.length;
    assert 3 == inObj[0].getData().getDimensions().length;
    if (!CuDNN.isEnabled()) return getCompatibilityLayer().eval(inObj);
    final TensorList inputData = inObj[0].getData();
    final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    final int[] outputDimensions = Arrays.copyOf(inputDimensions, 3);
    final int byteOffset = inputDimensions[1] * inputDimensions[0] * getFrom() * precision.size;
    outputDimensions[2] = getTo() - getFrom();
    return GpuHandle.run(nncontext -> {
      long size = (length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size);
      final CudaPtr cudaOutput = CudaPtr.allocate(nncontext.getDeviceNumber(), size, MemoryType.Managed, true);
      final CudaPtr cudaInput = CudaPtr.getCudaPtr(precision, inputData);
      final CudaResource<cudnnTensorDescriptor> inputDescriptor = getTensorDescriptor(inputDimensions, length, outputDimensions);
      final CudaResource<cudnnTensorDescriptor> outputDescriptor = getTensorDescriptor(outputDimensions, length, outputDimensions);
      CuDNN.cudnnTransformTensor(nncontext.getHandle(),
                                 precision.getPointer(1.0), inputDescriptor.getPtr(), cudaInput.getPtr().withByteOffset(byteOffset),
                                 precision.getPointer(0.0), outputDescriptor.getPtr(), cudaOutput.getPtr()
                                );
      final TensorList outputData = GpuTensorList.create(cudaOutput, length, outputDimensions, precision);
      //assert outputData.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      return getResult(inputData, inputDimensions, length, outputDimensions, byteOffset, outputData, inObj);
    });
  }
  
  public CudaResource<cudnnTensorDescriptor> getTensorDescriptor(int[] inputDimensions, int length, int[] outputDimensions) {
    return CuDNN.newTensorDescriptor(
      precision.code, length, outputDimensions[2], outputDimensions[1], outputDimensions[0], //
      inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
      inputDimensions[1] * inputDimensions[0], //
      inputDimensions[0], //
      1);
  }
  
  public NNResult getResult(TensorList inputData, int[] inputDimensions, int length, int[] outputDimensions, int byteOffset, TensorList outputData, NNResult[] inObj) {
    return new NNResult(outputData) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
      
      @Override
      protected void _accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
        if (!Arrays.equals(error.getDimensions(), outputData.getDimensions())) {
          throw new AssertionError(Arrays.toString(error.getDimensions()) + " != " + Arrays.toString(outputData.getDimensions()));
        }
        if (inObj[0].isAlive()) {
          final TensorList passbackTensorList = GpuHandle.run(nncontext -> {
            final CudaResource<cudnnTensorDescriptor> inputDescriptor = getTensorDescriptor(inputDimensions, length, outputDimensions);
            final CudaResource<cudnnTensorDescriptor> outputDescriptor = getTensorDescriptor(outputDimensions, length, outputDimensions);
            assert error.length() == inputData.length();
            //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
            final CudaPtr errorPtr = CudaPtr.getCudaPtr(precision, error);
            long size = (length * inputDimensions[2] * inputDimensions[1] * inputDimensions[0] * precision.size);
            final CudaPtr passbackBuffer = CudaPtr.allocate(nncontext.getDeviceNumber(), size, MemoryType.Managed, false);
            CuDNN.cudnnTransformTensor(nncontext.getHandle(),
                                       precision.getPointer(1.0), outputDescriptor.getPtr(), errorPtr.getPtr(),
                                       precision.getPointer(0.0), inputDescriptor.getPtr(), passbackBuffer.getPtr().withByteOffset(byteOffset)
                                      );
            return GpuTensorList.create(passbackBuffer, length, inputDimensions, precision);
            //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          });
          inObj[0].accumulate(buffer, passbackTensorList);
        }
        error.freeRef();
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
  
  /**
   * Gets to.
   *
   * @return the to
   */
  public int getTo() {
    return to;
  }
  
  /**
   * Sets to.
   *
   * @param to the to
   * @return the to
   */
  public ImgBandSelectLayer setTo(int to) {
    this.to = to;
    return this;
  }
}
