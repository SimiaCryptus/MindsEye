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

package com.simiacryptus.mindseye.layers.cudnn.f32;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.CudaResource;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CONVOLUTION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Schema output layer.
 */
public class SchemaOutputLayer extends NNLayer implements SchemaComponent {
  
  private final double logWeightInit;
  private final int inputBands;
  private final Map<String, double[]> features = new HashMap<>();
  /**
   * The Filter.
   */
  public Tensor filter;
  private String[] selected = {};
  
  /**
   * Instantiates a new Schema output layer.
   *
   * @param json the json
   */
  protected SchemaOutputLayer(JsonObject json) {
    super(json);
    this.logWeightInit = json.getAsJsonPrimitive("logWeightInit").getAsDouble();
    this.inputBands = json.getAsJsonPrimitive("inputBands").getAsInt();
    JsonObject featuresJson = json.getAsJsonObject("features");
    for (Map.Entry<String, JsonElement> e : featuresJson.entrySet()) {
      JsonArray jsonArray = e.getValue().getAsJsonArray();
      features.put(e.getKey(), IntStream.range(0, jsonArray.size()).mapToDouble(i -> jsonArray.get(i).getAsDouble()).toArray());
    }
    JsonArray selectedJson = json.getAsJsonArray("selected");
    setSchema(IntStream.range(0, selectedJson.size()).mapToObj(i -> selectedJson.get(i).getAsString()).toArray(i -> new String[i]));
  }
  
  /**
   * Instantiates a new Schema output layer.
   *
   * @param inputBands    the input bands
   * @param logWeightInit the log weight init
   */
  public SchemaOutputLayer(final int inputBands, final double logWeightInit) {
    super();
    this.inputBands = inputBands;
    this.logWeightInit = logWeightInit;
  }
  
  /**
   * From json schema output layer.
   *
   * @param json the json
   * @return the schema output layer
   */
  public static SchemaOutputLayer fromJson(JsonObject json) {
    return new SchemaOutputLayer(json);
  }
  
  public JsonObject getJson() {
    readFeatures();
    JsonObject json = super.getJsonStub();
    json.addProperty("inputBands", inputBands);
    json.addProperty("logWeightInit", logWeightInit);
    JsonArray jsonSelected = new JsonArray();
    for (String s : selected) jsonSelected.add(new JsonPrimitive(s));
    json.add("selected", jsonSelected);
    JsonObject jsonObject = new JsonObject();
    for (Map.Entry<String, double[]> e : features.entrySet()) {
      JsonArray valueArray = new JsonArray();
      for (double v : e.getValue()) valueArray.add(new JsonPrimitive(v));
      jsonObject.add(e.getKey(), valueArray);
    }
    json.add("features", jsonObject);
    return json;
  }
  
  @Override
  public SchemaOutputLayer setSchema(String... labels) {
    if (null == labels) throw new IllegalArgumentException();
    readFeatures();
    selected = labels;
    filter = new Tensor(1, 1, labels.length * inputBands);
    filter.fill(() -> (Math.random() - 0.5) * Math.pow(10, logWeightInit));
    for (int i = 0; i < labels.length; i++) {
      double[] feature = features.get(labels[i]);
      if (null == feature) continue;
      for (int j = 0; j < inputBands; j++) {
        filter.set(new int[]{0, 0, i * inputBands + j}, feature[j]);
      }
    }
    return this;
  }
  
  private void readFeatures() {
    if (null != filter && selected.length > 0) {
      for (int i = 0; i < selected.length; i++) {
        int offset = i * inputBands;
        double[] feature = new double[inputBands];
        for (int j = 0; j < inputBands; j++) {
          feature[j] = filter.get(0, 0, offset + j);
        }
        features.put(selected[i], feature);
      }
    }
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    ((CudaExecutionContext) nncontext).initThread();
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.getDimensions();
    assert (inputBands == inputSize[2]);
    int[] kernelSize = this.filter.getDimensions();
    int[] outputSize = getOutputSize(inputSize, kernelSize);
    int length = batch.length();
    
    try {
      
      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CudaResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
      CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, outputSize[2], outputSize[1], outputSize[0]);
      CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionDescriptor(
        0, 0, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
      final Pointer betaPtr = Pointer.to(new float[]{0.0f});
      final Pointer alphaPtr = Pointer.to(new float[]{1.0f});
      
      final float[] filterData = this.filter.getDataAsFloats();
      CudaPtr filterPtr = CuDNN.write(((CudaExecutionContext) nncontext).getDeviceNumber(), filterData);
      assert (0 < filterData.length);
      CudaPtr inputData = CudaPtr.toDeviceAsFloat(((CudaExecutionContext) nncontext).getDeviceNumber(), batch);
      assert kernelSize[0] * kernelSize[1] * kernelSize[2] == filterData.length;
      
      CudaPtr outputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.dim(outputSize) * 1l * length * Sizeof.FLOAT);
      try {
        assert verifyOutputDims(inputDescriptor, filterDescriptor, convolutionDescriptor, outputSize);
        int algorithm = ((CudaExecutionContext) nncontext).getForwardAlgorithm(
          inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
        CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateForwardWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
          inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
        CuDNN.handle(cudnnConvolutionForward(((CuDNN) nncontext).cudnnHandle, alphaPtr,
          inputDescriptor.getPtr(), inputData.getPtr(),
          filterDescriptor.getPtr(), filterPtr.getPtr(),
          convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, betaPtr,
          outputDescriptor.getPtr(), outputBuffer.getPtr()));
        workSpace.finalize();
      } catch (Throwable e) {
        throw new ComponentException("Error with " + Arrays.toString(kernelSize), e);
      }
      TensorList output = CudaPtr.fromDeviceFloat(outputBuffer, length, outputSize, ((CuDNN) nncontext).cudnnHandle);
      
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          outputBuffer.finalize();
          ((CudaExecutionContext) nncontext).initThread();
          assert (error.length() == batch.length());
          int length = error.length();
          CudaPtr errorPtr = CudaPtr.toDeviceAsFloat(((CudaExecutionContext) nncontext).getDeviceNumber(), error);
          if (!isFrozen()) {
            CudaPtr filterBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), filterData.length * 1l * Sizeof.FLOAT);
            try {
              int algorithm = ((CudaExecutionContext) nncontext).getBackwardFilterAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardFilterWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
              CuDNN.handle(cudnnConvolutionBackwardFilter(((CuDNN) nncontext).cudnnHandle,
                alphaPtr, inputDescriptor.getPtr(), inputData.getPtr(),
                outputDescriptor.getPtr(), errorPtr.getPtr(),
                convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size,
                betaPtr, filterDescriptor.getPtr(), filterBuffer.getPtr()));
              workSpace.finalize();
            } catch (Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(kernelSize), e);
            }
            final Tensor weightGradient = CudaPtr.fromDeviceFloat(filterBuffer, SchemaOutputLayer.this.filter.getDimensions());
            buffer.get(SchemaOutputLayer.this, SchemaOutputLayer.this.filter).accumulate(weightGradient.getData());
            filterBuffer.finalize();
          }
          if (input.isAlive()) {
            CudaPtr inputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.dim(batch.getDimensions()) * 1l * length * Sizeof.FLOAT);
            try {
              int algorithm = ((CudaExecutionContext) nncontext).getBackwardDataAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardDataWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
              CuDNN.handle(cudnnConvolutionBackwardData(((CuDNN) nncontext).cudnnHandle, alphaPtr,
                filterDescriptor.getPtr(), filterPtr.getPtr(),
                outputDescriptor.getPtr(), errorPtr.getPtr(),
                convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, betaPtr,
                inputDescriptor.getPtr(), inputBuffer.getPtr()));
              workSpace.finalize();
            } catch (Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(kernelSize), e);
            }
            TensorList inputBufferTensors = CudaPtr.fromDeviceFloat(inputBuffer, length, inputSize, ((CuDNN) nncontext).cudnnHandle);
            input.accumulate(buffer, inputBufferTensors);
            inputBuffer.finalize();
          }
          filterPtr.finalize();
        }
        
        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new ComponentException("Error mapCoords image with " + Arrays.toString(inputSize), e);
    }
  }
  
  /**
   * Get output size int [ ].
   *
   * @param inputSize  the input size
   * @param kernelSize the kernel size
   * @return the int [ ]
   */
  protected int[] getOutputSize(int[] inputSize, int[] kernelSize) {
    return IntStream.range(0, kernelSize.length).map(i -> {
      int x;
      if (i == kernelSize.length - 1) {
        x = kernelSize[i] / inputSize[i];
      }
      else {
        x = inputSize[i];
      }
      assert 0 < x;
      return x;
    }).toArray();
  }
  
  
  /**
   * Verify output dims boolean.
   *
   * @param inputDescriptor       the input descriptor
   * @param filterDescriptor      the filter descriptor
   * @param convolutionDescriptor the convolution descriptor
   * @param outputSize            the output size
   * @return the boolean
   */
  protected boolean verifyOutputDims(CudaResource<cudnnTensorDescriptor> inputDescriptor, CudaResource<cudnnFilterDescriptor> filterDescriptor, CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor, int[] outputSize) {
    int[] outputDims = CuDNN.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr());
    if (4 != outputDims.length) {
      return false;
    }
    if (outputSize[0] != outputDims[3]) {
      return false;
    }
    if (outputSize[1] != outputDims[2]) {
      return false;
    }
    return outputSize[2] == outputDims[1];
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.filter.getData());
  }
  
}
