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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Concatenates two or more inputs, assuming they have the same width and height, to produce an image with both inputs'
 * color bands. (e.g. Used in Inception modules in GoogLeNet.)
 */
@SuppressWarnings("serial")
public class ImgBandSelectLayer extends LayerBase implements MultiPrecision<ImgBandSelectLayer> {
  
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
  protected ImgBandSelectLayer(@javax.annotation.Nonnull final JsonObject json) {
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
  public static ImgBandSelectLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgBandSelectLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public Layer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer(IntStream.range(getFrom(), getTo()).toArray());
  }
  
  
  @Nullable
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    assert getFrom() < getTo();
    assert getFrom() >= 0;
    assert getTo() > 0;
    assert 1 == inObj.length;
    assert 3 == inObj[0].getData().getDimensions().length;
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final TensorList inputData = inObj[0].getData();
    @Nonnull final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    @javax.annotation.Nonnull final int[] outputDimensions = Arrays.copyOf(inputDimensions, 3);
    final int byteOffset = inputDimensions[1] * inputDimensions[0] * getFrom() * precision.size;
    outputDimensions[2] = getTo() - getFrom();
    long size = (length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size);
    return new Result(CudaSystem.eval(gpu -> {
      @javax.annotation.Nonnull final CudaMemory cudaOutput = gpu.allocate(size, MemoryType.Managed, true);
      @Nullable final CudaMemory cudaInput = gpu.getPtr(inputData, precision, MemoryType.Device);
      @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> inputDescriptor = getTensorDescriptor(inputDimensions, length, outputDimensions, gpu);
      @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> outputDescriptor = getTensorDescriptor(outputDimensions, length, outputDimensions, gpu);
      gpu.cudnnTransformTensor(
        precision.getPointer(1.0), inputDescriptor.getPtr(), cudaInput.getPtr().withByteOffset(byteOffset),
        precision.getPointer(0.0), outputDescriptor.getPtr(), cudaOutput.getPtr()
      );
      Arrays.stream(new ReferenceCounting[]{outputDescriptor, inputDescriptor, cudaInput}).forEach(ReferenceCounting::freeRef);
      return CudaTensorList.wrap(cudaOutput, length, outputDimensions, precision);
    }), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList error) -> {
      if (!Arrays.equals(error.getDimensions(), outputDimensions)) {
        throw new AssertionError(Arrays.toString(error.getDimensions()) + " != " + Arrays.toString(outputDimensions));
      }
      if (inObj[0].isAlive()) {
        final TensorList passbackTensorList = CudaSystem.eval(gpu -> {
          @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> inputDescriptor = getTensorDescriptor(inputDimensions, length, outputDimensions, gpu);
          @javax.annotation.Nonnull final CudaResource<cudnnTensorDescriptor> outputDescriptor = getTensorDescriptor(outputDimensions, length, outputDimensions, gpu);
          assert error.length() == inputData.length();
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
          @Nullable final CudaMemory errorPtr = gpu.getPtr(error, precision, MemoryType.Device);
          long size1 = (length * inputDimensions[2] * inputDimensions[1] * inputDimensions[0] * precision.size);
          @javax.annotation.Nonnull final CudaMemory passbackBuffer = gpu.allocate(size1, MemoryType.Managed, false);
          gpu.cudnnTransformTensor(
            precision.getPointer(1.0), outputDescriptor.getPtr(), errorPtr.getPtr(),
            precision.getPointer(0.0), inputDescriptor.getPtr(), passbackBuffer.getPtr().withByteOffset(byteOffset)
          );
          Arrays.stream(new ReferenceCounting[]{errorPtr, inputDescriptor, outputDescriptor}).forEach(ReferenceCounting::freeRef);
          return CudaTensorList.wrap(passbackBuffer, length, inputDimensions, precision);
          //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        });
        inObj[0].accumulate(buffer, passbackTensorList);
        passbackTensorList.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
      }
    };
  }
  
  /**
   * Gets tensor descriptor.
   *
   * @param inputDimensions  the input dimensions
   * @param length           the length
   * @param outputDimensions the output dimensions
   * @param deviceId         the device id
   * @return the tensor descriptor
   */
  @javax.annotation.Nonnull
  public CudaResource<cudnnTensorDescriptor> getTensorDescriptor(int[] inputDimensions, int length, int[] outputDimensions, final CudaDevice deviceId) {
    return deviceId.newTensorDescriptor(
      precision.code, length, outputDimensions[2], outputDimensions[1], outputDimensions[0], //
      inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
      inputDimensions[1] * inputDimensions[0], //
      inputDimensions[0], //
      1);
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
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
  @javax.annotation.Nonnull
  public ImgBandSelectLayer setFrom(final int from) {
    this.from = from;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public ImgBandSelectLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @javax.annotation.Nonnull
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
  @javax.annotation.Nonnull
  public ImgBandSelectLayer setTo(int to) {
    this.to = to;
    return this;
  }
}
