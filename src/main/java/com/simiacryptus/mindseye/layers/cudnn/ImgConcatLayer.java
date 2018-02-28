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
public class ImgConcatLayer extends LayerBase implements MultiPrecision<ImgConcatLayer> {
  
  private int maxBands = -1;
  private Precision precision = Precision.Double;
  private boolean parallel = true;
  
  /**
   * Instantiates a new Img concat layer.
   */
  public ImgConcatLayer() {
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   */
  protected ImgConcatLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    maxBands = json.get("maxBands").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
    this.parallel = json.get("parallel").getAsBoolean();
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static ImgConcatLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgConcatLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgConcatLayer.class);
  }
  
  
  @Nullable
  @Override
  public Result evalAndFree(@javax.annotation.Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    assert 3 == inObj[0].getData().getDimensions().length;
    @Nonnull final int[] outputDimensions = inObj[0].getData().getDimensions();
    final int length = inObj[0].getData().length();
    assert Arrays.stream(inObj).allMatch(x -> {
      @Nonnull int[] d = x.getData().getDimensions();
      return 3 == d.length && d[0] == outputDimensions[0] && d[1] == outputDimensions[1] && x.getData().length() == length;
    });
    outputDimensions[2] = Arrays.stream(inObj).mapToInt(x -> x.getData().getDimensions()[2]).sum();
    if (0 < maxBands && outputDimensions[2] > maxBands) {
      outputDimensions[2] = maxBands;
    }
    return new Result(CudaSystem.eval(gpu -> {
      final long outputSize = ((long) length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size);
      @javax.annotation.Nonnull final CudaMemory cudaOutput = gpu.allocate(outputSize, MemoryType.Managed, true);
      IntStream stream = IntStream.range(0, inObj.length);
      //if (!CoreSettings.INSTANCE.isConservative() && parallel) stream = stream.parallel();
      stream.forEach(i -> {
        final TensorList input = inObj[i].getData();
        @Nonnull final int[] inputDimensions = input.getDimensions();
        assert inputDimensions[0] == outputDimensions[0];
        assert inputDimensions[1] == outputDimensions[1];
        int bandOffset = IntStream.range(0, i).map(j -> inObj[j].getData().getDimensions()[2]).sum();
        if (maxBands > 0) bandOffset = Math.min(bandOffset, maxBands);
        int inputBands = inputDimensions[2];
        if (maxBands > 0) inputBands = Math.min(inputBands, maxBands - bandOffset);
        if (inputBands > 0) {
          @Nullable final CudaTensor cudaInput = gpu.getTensor(input, precision, MemoryType.Device);
          assert inputBands > 0;
          assert maxBands <= 0 || inputBands <= maxBands;
          assert inputBands <= inputDimensions[2];
          @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = getTensorDescriptor(length, inputBands, inputDimensions, outputDimensions, gpu);
          
          @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(
            precision.code, length, inputBands, inputDimensions[1], inputDimensions[0], //
            cudaInput.descriptor.nStride, //
            cudaInput.descriptor.cStride, //
            cudaInput.descriptor.hStride, //
            cudaInput.descriptor.wStride);
//          @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = getTensorDescriptor(length, inputBands, inputDimensions, inputDimensions, gpu);
          
          int byteOffset = inputDimensions[1] * inputDimensions[0] * bandOffset * precision.size;
          gpu.cudnnTransformTensor(
            precision.getPointer(1.0), inputDescriptor.getPtr(), cudaInput.memory.getPtr(),
            precision.getPointer(0.0), outputDescriptor.getPtr(), cudaOutput.getPtr().withByteOffset(byteOffset)
          );
          Arrays.stream(new ReferenceCounting[]{cudaInput, outputDescriptor, inputDescriptor}).forEach(ReferenceCounting::freeRef);
        }
      });
      CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(precision.code, length, outputDimensions[2], outputDimensions[1], outputDimensions[0]);
      return CudaTensorList.wrap(CudaTensor.wrap(cudaOutput, outDesc), length, outputDimensions, precision);
    }), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      if (!Arrays.equals(delta.getDimensions(), outputDimensions)) {
        throw new AssertionError(Arrays.toString(delta.getDimensions()) + " != " + Arrays.toString(outputDimensions));
      }
      //outputBuffer.freeRef();
      assert delta.length() == inObj[0].getData().length();
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
      @javax.annotation.Nonnull IntStream stream = IntStream.range(0, inObj.length);
      if (!CoreSettings.INSTANCE.isConservative() && parallel) stream = stream.parallel();
      stream.forEach(i -> {
        final Result input = inObj[i];
        @Nonnull int[] inputDimensions = input.getData().getDimensions();
        assert 3 == inputDimensions.length;
        assert delta.length() == input.getData().length();
        assert inputDimensions[0] == outputDimensions[0];
        assert inputDimensions[1] == outputDimensions[1];
        int bandOffset = IntStream.range(0, i).map(j -> inObj[j].getData().getDimensions()[2]).sum();
        int inputBands = maxBands <= 0 ? inputDimensions[2] : Math.min(inputDimensions[2], maxBands - bandOffset);
        if (inputBands > 0 && input.isAlive()) {
          assert inputBands <= inputDimensions[2];
          final TensorList passbackTensorList = CudaSystem.eval(gpu -> {
            @javax.annotation.Nonnull int[] viewDimensions = Arrays.copyOf(inputDimensions, inputDimensions.length);
            viewDimensions[2] = inputBands;
            @Nullable final CudaTensor cudaDelta = gpu.getTensor(delta, precision, MemoryType.Device);
            long inputSize = (length * inputDimensions[2] * inputDimensions[1] * inputDimensions[0] * precision.size);
            @javax.annotation.Nonnull final CudaMemory cudaBackprop = gpu.allocate(inputSize, MemoryType.Managed, (inputBands + bandOffset) > outputDimensions[2]);
            @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = getTensorDescriptor(length, inputBands, viewDimensions, inputDimensions, gpu);
            @javax.annotation.Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = getTensorDescriptor(length, inputBands, viewDimensions, outputDimensions, gpu);
            int byteOffset = outputDimensions[1] * outputDimensions[0] * bandOffset * precision.size;
            gpu.cudnnTransformTensor(
              precision.getPointer(1.0), outputDescriptor.getPtr(), cudaDelta.memory.getPtr().withByteOffset(byteOffset),
              precision.getPointer(0.0), inputDescriptor.getPtr(), cudaBackprop.getPtr()
            );
            Arrays.stream(new ReferenceCounting[]{cudaDelta, outputDescriptor}).forEach(ReferenceCounting::freeRef);
            return CudaTensorList.wrap(CudaTensor.wrap(cudaBackprop, inputDescriptor), length, inputDimensions, precision);
          });
          input.accumulate(buffer, passbackTensorList);
        }
        //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      });
    }) {
      
      @Override
      protected void _free() {
        for (@javax.annotation.Nonnull Result result : inObj) {
          result.freeRef();
          result.getData().freeRef();
        }
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
   * @param length           the length
   * @param inputBands       the input bands
   * @param viewXY  the input dimensions
   * @param sizeDimensions the output dimensions
   * @param deviceId         the device id
   * @return the tensor descriptor
   */
  @javax.annotation.Nonnull
  public CudaDevice.CudaTensorDescriptor getTensorDescriptor(int length, int inputBands, int[] viewXY, int[] sizeDimensions, final CudaDevice deviceId) {
    return deviceId.newTensorDescriptor(
      precision.code, length, inputBands, viewXY[1], viewXY[0], //
      sizeDimensions[2] * sizeDimensions[1] * sizeDimensions[0], //
      sizeDimensions[1] * sizeDimensions[0], //
      sizeDimensions[0], //
      1);
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("maxBands", maxBands);
    json.addProperty("precision", precision.name());
    json.addProperty("parallel", isParallel());
    return json;
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
  @javax.annotation.Nonnull
  public ImgConcatLayer setMaxBands(final int maxBands) {
    this.maxBands = maxBands;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public ImgConcatLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  
  /**
   * Is parallel boolean.
   *
   * @return the boolean
   */
  public boolean isParallel() {
    return parallel;
  }
  
  /**
   * Sets parallel.
   *
   * @param parallel the parallel
   * @return the parallel
   */
  public ImgConcatLayer setParallel(boolean parallel) {
    this.parallel = parallel;
    return this;
  }
  
}
