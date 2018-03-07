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
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaDevice;
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.Precision;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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
  protected ImgConcatLayer(@Nonnull final JsonObject json) {
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
  public static ImgConcatLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgConcatLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgConcatLayer.class);
  }
  
  
  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    int[] dimensions = inObj[0].getData().getDimensions();
    assert 3 == dimensions.length;
    @Nonnull final int[] outputDimensions = Arrays.copyOf(dimensions, dimensions.length);
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
      @Nonnull final CudaMemory cudaOutput = gpu.allocate(outputSize, MemoryType.Managed, true);
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
          @Nullable final CudaTensor cudaInput = gpu.getTensor(input, precision, MemoryType.Device, false);
          assert inputBands > 0;
          assert maxBands <= 0 || inputBands <= maxBands;
          assert inputBands <= inputDimensions[2];
          @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(
            precision, length, inputBands, outputDimensions[1], outputDimensions[0], //
            outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
            outputDimensions[1] * outputDimensions[0], //
            outputDimensions[0], //
            1);
  
          @Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(
            precision, length, inputBands, inputDimensions[1], inputDimensions[0], //
            cudaInput.descriptor.nStride, //
            cudaInput.descriptor.cStride, //
            cudaInput.descriptor.hStride, //
            cudaInput.descriptor.wStride);
  
          int byteOffset = outputDescriptor.cStride * bandOffset * precision.size;
          CudaMemory cudaInputMemory = cudaInput.getMemory(gpu);
          gpu.cudnnTransformTensor(
            precision.getPointer(1.0), inputDescriptor.getPtr(), cudaInputMemory.getPtr(),
            precision.getPointer(0.0), outputDescriptor.getPtr(), cudaOutput.getPtr().withByteOffset(byteOffset)
          );
          cudaInputMemory.freeRef();
          Stream.<ReferenceCounting>of(cudaInput, outputDescriptor, inputDescriptor).forEach(ReferenceCounting::freeRef);
        }
      });
      CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(precision, length, outputDimensions[2], outputDimensions[1], outputDimensions[0]);
      return CudaTensorList.wrap(CudaTensor.wrap(cudaOutput, outDesc, precision), length, outputDimensions, precision);
    }, Arrays.stream(inObj).map(Result::getData).toArray()), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      assert delta.getDimensions()[0] == outputDimensions[0];
      assert delta.getDimensions()[1] == outputDimensions[1];
      assert delta.getDimensions()[2] == outputDimensions[2];
      if (!Arrays.equals(delta.getDimensions(), outputDimensions)) {
        throw new AssertionError(Arrays.toString(delta.getDimensions()) + " != " + Arrays.toString(outputDimensions));
      }
      //outputBuffer.freeRef();
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
      @Nonnull IntStream stream = IntStream.range(0, inObj.length);
      if (!CoreSettings.INSTANCE.isSingleThreaded() && parallel) stream = stream.parallel();
      stream.forEach(i -> {
        final Result input = inObj[i];
        int[] inputDimentions = input.getData().getDimensions();
        assert 3 == inputDimentions.length;
        assert delta.length() == input.getData().length();
        assert inputDimentions[0] == outputDimensions[0];
        assert inputDimentions[1] == outputDimensions[1];
        int bandOffset = IntStream.range(0, i).map(j -> inObj[j].getData().getDimensions()[2]).sum();
        int inputBands = maxBands <= 0 ? inputDimentions[2] : Math.min(inputDimentions[2], maxBands - bandOffset);
        if (inputBands > 0 && input.isAlive()) {
          assert inputBands <= inputDimentions[2];
          assert inputBands <= outputDimensions[2];
          final TensorList passbackTensorList = CudaSystem.eval(gpu -> {
            final CudaTensor result;
            synchronized (gpu) {result = gpu.getTensor(delta, precision, MemoryType.Device, true);}
            @Nullable final CudaTensor cudaDelta = result;
            CudaMemory cudaDeltaMemory = cudaDelta.getMemory(gpu);
            try {
              if (inputDimentions[2] == inputBands) {
                @Nonnull final CudaDevice.CudaTensorDescriptor viewDescriptor = gpu.newTensorDescriptor(
                  precision, length, inputDimentions[2], inputDimentions[1], inputDimentions[0], //
                  cudaDelta.descriptor.nStride, //
                  cudaDelta.descriptor.cStride, //
                  cudaDelta.descriptor.hStride, //
                  cudaDelta.descriptor.wStride);
                int byteOffset = cudaDelta.descriptor.cStride * bandOffset * precision.size;
                CudaMemory ptr = cudaDeltaMemory.withByteOffset(byteOffset);
                CudaTensor cudaTensor = CudaTensor.wrap(ptr, viewDescriptor, precision);
                Stream.<ReferenceCounting>of(cudaDelta).forEach(ReferenceCounting::freeRef);
                return CudaTensorList.wrap(cudaTensor, length, inputDimentions, precision);
              }
              else {
                @Nonnull final CudaDevice.CudaTensorDescriptor passbackTransferDescriptor = gpu.newTensorDescriptor(
                  precision, length, inputBands, inputDimentions[1], inputDimentions[0], //
                  inputDimentions[2] * inputDimentions[1] * inputDimentions[0], //
                  inputDimentions[1] * inputDimentions[0], //
                  inputDimentions[0], //
                  1);
                @Nonnull final CudaDevice.CudaTensorDescriptor passbackDescriptor = gpu.newTensorDescriptor(
                  precision, length, inputDimentions[2], inputDimentions[1], inputDimentions[0], //
                  inputDimentions[2] * inputDimentions[1] * inputDimentions[0], //
                  inputDimentions[1] * inputDimentions[0], //
                  inputDimentions[0], //
                  1);
                @Nonnull final CudaDevice.CudaTensorDescriptor deltaViewDescriptor = gpu.newTensorDescriptor(
                  precision, length, inputBands, inputDimentions[1], inputDimentions[0], //
                  cudaDelta.descriptor.nStride, //
                  cudaDelta.descriptor.cStride, //
                  cudaDelta.descriptor.hStride, //
                  cudaDelta.descriptor.wStride);
                @Nonnull final CudaMemory cudaBackprop = gpu.allocate(
                  (long) passbackDescriptor.nStride * length * precision.size,
                  MemoryType.Managed, inputBands == inputDimentions[2]);
                int byteOffset = cudaDelta.descriptor.cStride * bandOffset * precision.size;
                gpu.cudnnTransformTensor(
                  precision.getPointer(1.0), deltaViewDescriptor.getPtr(), cudaDeltaMemory.getPtr().withByteOffset(byteOffset),
                  precision.getPointer(0.0), passbackTransferDescriptor.getPtr(), cudaBackprop.getPtr()
                );
                Stream.<ReferenceCounting>of(cudaDelta, deltaViewDescriptor, passbackTransferDescriptor).forEach(ReferenceCounting::freeRef);
                return CudaTensorList.wrap(CudaTensor.wrap(cudaBackprop, passbackDescriptor, precision), length, inputDimentions, precision);
              }
            } finally {
              cudaDeltaMemory.freeRef();
            }
          });
          input.accumulate(buffer, passbackTensorList);
        }
        //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      });
    }) {
      
      @Override
      protected void _free() {
        for (@Nonnull Result result : inObj) {
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
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
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
  @Nonnull
  public ImgConcatLayer setMaxBands(final int maxBands) {
    this.maxBands = maxBands;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public ImgConcatLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Nonnull
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
