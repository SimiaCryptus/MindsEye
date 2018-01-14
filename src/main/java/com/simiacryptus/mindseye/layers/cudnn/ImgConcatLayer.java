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
import com.simiacryptus.mindseye.layers.cudnn.lang.*;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Concatenates two or more inputs, assuming they have the same width and height, to produce an image with both inputs'
 * color bands. (e.g. Used in Inception modules in GoogLeNet.)
 */
@SuppressWarnings("serial")
public class ImgConcatLayer extends NNLayer implements LayerPrecision<ImgConcatLayer> {
  
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
    if (0 == CuDNN.gpuContexts.size()) return getCompatibilityLayer().eval(inObj);
    return CuDNN.gpuContexts.run(nncontext -> {
      if (nncontext.getDeviceNumber() < 0) return getCompatibilityLayer().eval(inObj);
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
      nncontext.initThread();
      final CudaPtr cudaOutput = CuDNN.alloc(nncontext.getDeviceNumber(), length * outputDimensions[2] * outputDimensions[1] * outputDimensions[0] * precision.size, true);
      int bandOffset = 0;
      for (int i = 0; i < inObj.length; i++) {
        final TensorList input = inObj[i].getData();
        final CudaPtr cudaInput = CudaPtr.write(nncontext.getDeviceNumber(), precision, input);
        final int[] inputDimensions = input.getDimensions();
        assert inputDimensions[0] == outputDimensions[0];
        assert inputDimensions[1] == outputDimensions[1];
        final int inputBands = maxBands <= 0 ? inputDimensions[2] : Math.min(inputDimensions[2], maxBands - bandOffset);
        if (inputBands > 0) {
          assert inputBands > 0;
          assert inputBands <= maxBands;
          assert inputBands <= inputDimensions[2];
        
          final CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
            precision.code, length, inputBands, inputDimensions[1], inputDimensions[0],
            inputDimensions[2] * inputDimensions[1] * inputDimensions[0],//
            inputDimensions[1] * inputDimensions[0],//
            inputDimensions[0],//
            1);
          final CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
            precision.code, length, inputBands, inputDimensions[1], inputDimensions[0],
            outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
            outputDimensions[1] * outputDimensions[0], //
            outputDimensions[0], //
            1);
          CuDNN.cudnnTransformTensor(nncontext.cudnnHandle,
                                     precision.getPointer(1.0), inputDescriptor.getPtr(), cudaInput.getPtr(),
                                     precision.getPointer(0.0), outputDescriptor.getPtr(), cudaOutput.getPtr().withByteOffset(inputDimensions[1] * inputDimensions[0] * bandOffset * precision.size)
                                    );
          bandOffset += inputBands;
        }
      }
      final TensorList outputData = new GpuTensorList(cudaOutput, length, outputDimensions, nncontext.cudnnHandle, precision);
      //assert outputData.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      return new NNResult(outputData) {
      
        @Override
        public void free() {
          Arrays.stream(inObj).forEach(NNResult::free);
          cudaOutput.finalize();
        }
      
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList delta) {
          CuDNN.gpuContexts.apply(nncontext -> {
            if (!Arrays.equals(delta.getDimensions(), outputData.getDimensions())) {
              throw new AssertionError(Arrays.toString(delta.getDimensions()) + " != " + Arrays.toString(outputData.getDimensions()));
            }
            //outputBuffer.free();
            nncontext.initThread();
            assert delta.length() == inObj[0].getData().length();
            //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
            int deviceNumber = nncontext.getDeviceNumber();
            final CudaPtr cudaDelta = CudaPtr.write(deviceNumber, precision, delta);
            int bandOffset = 0;
            for (int i = 0; i < inObj.length; i++) {
              final NNResult input = inObj[i];
              assert 3 == input.getData().getDimensions().length;
              assert delta.length() == input.getData().length();
              final int[] inputDimensions = input.getData().getDimensions();
              assert inputDimensions[0] == outputDimensions[0];
              assert inputDimensions[1] == outputDimensions[1];
              final int inputBands = maxBands <= 0 ? inputDimensions[2] : Math.min(inputDimensions[2], maxBands - bandOffset);
              if (inputBands > 0 && input.isAlive()) {
                assert inputBands <= inputDimensions[2];
                final CudaPtr cudaBackprop = CuDNN.alloc(deviceNumber, length * inputDimensions[2] * inputDimensions[1] * inputDimensions[0] * precision.size, false);
                final CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
                  precision.code, length, inputBands, inputDimensions[1], inputDimensions[0], //
                  inputDimensions[2] * inputDimensions[1] * inputDimensions[0], //
                  inputDimensions[1] * inputDimensions[0], //
                  inputDimensions[0], //
                  1);
                final CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
                  precision.code, length, inputBands, inputDimensions[1], inputDimensions[0], //
                  outputDimensions[2] * outputDimensions[1] * outputDimensions[0], //
                  outputDimensions[1] * outputDimensions[0], //
                  outputDimensions[0], //
                  1);
                CuDNN.cudnnTransformTensor(nncontext.cudnnHandle,
                                           precision.getPointer(1.0), outputDescriptor.getPtr(), cudaDelta.getPtr().withByteOffset(outputDimensions[1] * outputDimensions[0] * bandOffset * precision.size),
                                           precision.getPointer(0.0), inputDescriptor.getPtr(), cudaBackprop.getPtr()
                                          );
                final TensorList passbackTensorList = new GpuTensorList(cudaBackprop, length, inputDimensions, nncontext.cudnnHandle, precision);
                //assert passbackTensorList.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
                input.accumulate(buffer, passbackTensorList);
                cudaBackprop.finalize();
              }
              bandOffset += inputBands;
            }
          });
        }
      
        @Override
        public boolean isAlive() {
          return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
        }
      };
    });
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
