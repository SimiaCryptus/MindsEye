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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaDevice;
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Concatenates two or more inputs, assuming they have the same width and height, to produce an image apply both inputs'
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
  protected ImgBandSelectLayer(@Nonnull final JsonObject json) {
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
  public static ImgBandSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandSelectLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer(IntStream.range(getFrom(), getTo()).toArray());
  }
  
  
  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    assert getFrom() < getTo();
    assert getFrom() >= 0;
    assert getTo() > 0;
    assert 1 == inObj.length;
    assert 3 == inObj[0].getData().getDimensions().length;
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    final TensorList inputData = inObj[0].getData();
    @Nonnull final int[] inputDimensions = inputData.getDimensions();
    final int length = inputData.length();
    @Nonnull final int[] outputDimensions = Arrays.copyOf(inputDimensions, 3);
    outputDimensions[2] = getTo() - getFrom();
    long size = (length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size);
    return new Result(CudaSystem.run(gpu -> {
      @Nullable final CudaTensor cudaInput = gpu.getTensor(inputData, precision, MemoryType.Device, false);
      inputData.freeRef();
      final int byteOffset = cudaInput.descriptor.cStride * getFrom() * precision.size;
      @Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(
        precision, length, outputDimensions[2], outputDimensions[1], outputDimensions[0], //
        cudaInput.descriptor.nStride, //
        cudaInput.descriptor.cStride, //
        cudaInput.descriptor.hStride, //
        cudaInput.descriptor.wStride);
      CudaMemory cudaInputMemory = cudaInput.getMemory(gpu);
      assert CudaDevice.isThreadDeviceId(gpu.getDeviceId());
      CudaTensor cudaTensor = CudaTensor.wrap(cudaInputMemory.withByteOffset(byteOffset), inputDescriptor, precision);
      Stream.<ReferenceCounting>of(cudaInput, cudaInputMemory).forEach(ReferenceCounting::freeRef);
      return CudaTensorList.wrap(cudaTensor, length, outputDimensions, precision);
    }, inputData), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (!Arrays.equals(delta.getDimensions(), outputDimensions)) {
        throw new AssertionError(Arrays.toString(delta.getDimensions()) + " != " + Arrays.toString(outputDimensions));
      }
      if (inObj[0].isAlive()) {
        final TensorList passbackTensorList = CudaSystem.run(gpu -> {
          @Nonnull final CudaDevice.CudaTensorDescriptor viewDescriptor = gpu.newTensorDescriptor(
            precision, length, outputDimensions[2], outputDimensions[1], outputDimensions[0], //
            inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
            inputDimensions[1] * inputDimensions[0], //
            inputDimensions[0], //
            1);
          @Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(
            precision, length, inputDimensions[2], inputDimensions[1], inputDimensions[0], //
            inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
            inputDimensions[1] * inputDimensions[0], //
            inputDimensions[0], //
            1);
          final int byteOffset = viewDescriptor.cStride * getFrom() * precision.size;
          assert delta.length() == length;
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
          @Nullable final CudaTensor errorPtr = gpu.getTensor(delta, precision, MemoryType.Device, false);
          delta.freeRef();
          long size1 = (length * inputDimensions[2] * inputDimensions[1] * inputDimensions[0] * precision.size);
          @Nonnull final CudaMemory passbackBuffer = gpu.allocate(size1, MemoryType.Managed.normalize(), false);
          CudaMemory errorPtrMemory = errorPtr.getMemory(gpu);
          gpu.cudnnTransformTensor(
            precision.getPointer(1.0), errorPtr.descriptor.getPtr(), errorPtrMemory.getPtr(),
            precision.getPointer(0.0), viewDescriptor.getPtr(), passbackBuffer.getPtr().withByteOffset(byteOffset)
          );
          errorPtrMemory.dirty();
          passbackBuffer.dirty();
          errorPtrMemory.freeRef();
          CudaTensor cudaTensor = CudaTensor.wrap(passbackBuffer, inputDescriptor, precision);
          Stream.<ReferenceCounting>of(errorPtr, viewDescriptor).forEach(ReferenceCounting::freeRef);
          return CudaTensorList.wrap(cudaTensor, length, inputDimensions, precision);
          //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        }, delta);
        inObj[0].accumulate(buffer, passbackTensorList);
      }
      else {
        delta.freeRef();
      }
    }) {
  
      @Override
      public void accumulate(final DeltaSet<Layer> buffer, final TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }
  
  
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
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
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
  @Nonnull
  public ImgBandSelectLayer setFrom(final int from) {
    this.from = from;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public ImgBandSelectLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Nonnull
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
  @Nonnull
  public ImgBandSelectLayer setTo(int to) {
    this.to = to;
    return this;
  }
}
