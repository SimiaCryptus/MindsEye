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
public class ImgConcatLayer extends NNLayer implements MultiPrecision<ImgConcatLayer> {
  
  private int maxBands = -1;
  private Precision precision = Precision.Double;
  
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
  protected ImgConcatLayer(final JsonObject json) {
    super(json);
    maxBands = json.get("maxBands").getAsInt();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static ImgConcatLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ImgConcatLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public NNLayer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgConcatLayer.class);
  }
  
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    if (!GpuSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    assert 3 == inObj[0].getData().getDimensions().length;
    final int[] outputDimensions = inObj[0].getData().getDimensions();
    final int length = inObj[0].getData().length();
    assert Arrays.stream(inObj).allMatch(x -> {
      int[] d = x.getData().getDimensions();
      return 3 == d.length && d[0] == outputDimensions[0] && d[1] == outputDimensions[1] && x.getData().length() == length;
    });
    outputDimensions[2] = Arrays.stream(inObj).mapToInt(x -> x.getData().getDimensions()[2]).sum();
    if (0 < maxBands && outputDimensions[2] > maxBands) {
      outputDimensions[2] = maxBands;
    }
    for (NNResult nnResult : inObj) {
      nnResult.addRef();
      nnResult.getData().addRef();
    }
    return new NNResult(CuDNNHandle.run(gpu -> {
      final long outputSize = (length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size);
      final CudaPtr cudaOutput = CudaPtr.allocate(gpu.getDeviceNumber(), outputSize, MemoryType.Managed, true);
      for (int i = 0; i < inObj.length; i++) {
        final TensorList input = inObj[i].getData();
        final int[] inputDimensions = input.getDimensions();
        assert inputDimensions[0] == outputDimensions[0];
        assert inputDimensions[1] == outputDimensions[1];
        int bandOffset = IntStream.range(0, i).map(j -> inObj[j].getData().getDimensions()[2]).sum();
        if (maxBands > 0) bandOffset = Math.min(bandOffset, maxBands);
        int inputBands = inputDimensions[2];
        if (maxBands > 0) inputBands = Math.min(inputBands, maxBands - bandOffset);
        if (inputBands > 0) {
          final CudaPtr cudaInput = CudaPtr.getCudaPtr(precision, input);
          assert inputBands > 0;
          assert maxBands <= 0 || inputBands <= maxBands;
          assert inputBands <= inputDimensions[2];
          final CudaResource<cudnnTensorDescriptor> inputDescriptor = getTensorDescriptor(length, inputBands, inputDimensions, inputDimensions);
          final CudaResource<cudnnTensorDescriptor> outputDescriptor = getTensorDescriptor(length, inputBands, inputDimensions, outputDimensions);
          int byteOffset = inputDimensions[1] * inputDimensions[0] * bandOffset * precision.size;
          CuDNNHandle.cudnnTransformTensor(gpu.getHandle(),
                                           precision.getPointer(1.0), inputDescriptor.getPtr(), cudaInput.getPtr(),
                                           precision.getPointer(0.0), outputDescriptor.getPtr(), cudaOutput.getPtr().withByteOffset(byteOffset)
                                          );
          gpu.registerForCleanup(cudaInput);
        }
      }
      return GpuTensorList.wrap(cudaOutput, length, outputDimensions, precision);
    }), (final DeltaSet<NNLayer> buffer, final TensorList delta) -> {
      if (!Arrays.equals(delta.getDimensions(), outputDimensions)) {
        throw new AssertionError(Arrays.toString(delta.getDimensions()) + " != " + Arrays.toString(outputDimensions));
      }
      //outputBuffer.freeRef();
      assert delta.length() == inObj[0].getData().length();
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
      IntStream.range(0, inObj.length).parallel().forEach(i -> {
        final NNResult input = inObj[i];
        int[] inputDimensions = input.getData().getDimensions();
        assert 3 == inputDimensions.length;
        assert delta.length() == input.getData().length();
        assert inputDimensions[0] == outputDimensions[0];
        assert inputDimensions[1] == outputDimensions[1];
        int bandOffset1 = IntStream.range(0, i).map(j -> inObj[j].getData().getDimensions()[2]).sum();
        int inputBands1 = maxBands <= 0 ? inputDimensions[2] : Math.min(inputDimensions[2], maxBands - bandOffset1);
        if (inputBands1 > 0 && input.isAlive()) {
          assert inputBands1 <= inputDimensions[2];
          final TensorList passbackTensorList = CuDNNHandle.run(gpu -> {
            final CudaPtr cudaDelta = CudaPtr.getCudaPtr(precision, delta);
            long inputSize = (length * inputDimensions[2] * inputDimensions[1] * inputDimensions[0] * precision.size);
            final CudaPtr cudaBackprop = CudaPtr.allocate(gpu.getDeviceNumber(), inputSize, MemoryType.Managed, true);
            final CudaResource<cudnnTensorDescriptor> inputDescriptor = getTensorDescriptor(length, inputBands1, inputDimensions, inputDimensions);
            final CudaResource<cudnnTensorDescriptor> outputDescriptor = getTensorDescriptor(length, inputBands1, inputDimensions, outputDimensions);
            int byteOffset1 = outputDimensions[1] * outputDimensions[0] * bandOffset1 * precision.size;
            CuDNNHandle.cudnnTransformTensor(gpu.getHandle(),
                                             precision.getPointer(1.0), outputDescriptor.getPtr(), cudaDelta.getPtr().withByteOffset(byteOffset1),
                                             precision.getPointer(0.0), inputDescriptor.getPtr(), cudaBackprop.getPtr()
                                            );
            gpu.registerForCleanup(cudaDelta);
            return GpuTensorList.wrap(cudaBackprop, length, inputDimensions, precision);
          });
          input.accumulate(buffer, passbackTensorList);
          passbackTensorList.freeRef();
        }
        //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      });
    }) {
      
      @Override
      protected void _free() {
        for (NNResult nnResult : inObj) {
          nnResult.freeRef();
          nnResult.getData().freeRef();
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
   * @param imageDimensions  the input dimensions
   * @param strideDimensions the output dimensions
   * @return the tensor descriptor
   */
  public CudaResource<cudnnTensorDescriptor> getTensorDescriptor(int length, int inputBands, int[] imageDimensions, int[] strideDimensions) {
    return GpuSystem.newTensorDescriptor(
      precision.code, length, inputBands, imageDimensions[1], imageDimensions[0], //
      strideDimensions[2] * strideDimensions[1] * strideDimensions[0], //
      strideDimensions[1] * strideDimensions[0], //
      strideDimensions[0], //
      1);
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.addProperty("maxBands", maxBands);
    json.addProperty("precision", precision.name());
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
  public ImgConcatLayer setMaxBands(final int maxBands) {
    this.maxBands = maxBands;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public ImgConcatLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
