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
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.CudaResource;
import com.simiacryptus.util.ml.Tensor;
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
 * The type Convolution layer.
 */
public class SchemaOutputLayer extends NNLayer implements SchemaComponent {
  
  public JsonObject getJson() {
    readFeatures();
    JsonObject json = super.getJsonStub();
    json.addProperty("inputBands", inputBands);
    json.addProperty("logWeightInit", logWeightInit);
    JsonArray jsonSelected = new JsonArray();
    for(String s : selected) jsonSelected.add(new JsonPrimitive(s));
    json.add("selected", jsonSelected);
    JsonObject jsonObject = new JsonObject();
    for(Map.Entry<String, double[]> e : features.entrySet()) {
      JsonArray valueArray = new JsonArray();
      for(double v : e.getValue()) valueArray.add(new JsonPrimitive(v));
      jsonObject.add(e.getKey(), valueArray);
    }
    json.add("features", jsonObject);
    return json;
  }
  
  public static SchemaOutputLayer fromJson(JsonObject json) {
    return new SchemaOutputLayer(json);
  }
  
  protected SchemaOutputLayer(JsonObject json) {
    super(json);
    this.logWeightInit = json.getAsJsonPrimitive("logWeightInit").getAsDouble();
    this.inputBands = json.getAsJsonPrimitive("inputBands").getAsInt();
    JsonObject featuresJson = json.getAsJsonObject("features");
    for(Map.Entry<String, JsonElement> e : featuresJson.entrySet()) {
      JsonArray jsonArray = e.getValue().getAsJsonArray();
      features.put(e.getKey(), IntStream.range(0, jsonArray.size()).mapToDouble(i -> jsonArray.get(i).getAsDouble()).toArray());
    }
    JsonArray selectedJson = json.getAsJsonArray("selected");
    setSchema(IntStream.range(0,selectedJson.size()).mapToObj(i->selectedJson.get(i).getAsString()).toArray(i->new String[i]));
  }
  
  
  /**
   * The Filter.
   */
  public Tensor filter;
  private final double logWeightInit;
  private final int inputBands;
  private String[] selected = new String[]{};
  private Map<String,double[]> features = new HashMap<>();
  
  public SchemaOutputLayer(final int inputBands, final double logWeightInit) {
    super();
    this.inputBands = inputBands;
    this.logWeightInit = logWeightInit;
  }
  
  @Override
  public SchemaOutputLayer setSchema(String... labels) {
    if(null == labels) throw new RuntimeException();
    readFeatures();
    selected = labels;
    filter = new Tensor(1,1,labels.length*inputBands);
    filter.fill(()->(Math.random()-0.5)*Math.pow(10,logWeightInit));
    for(int i=0;i<labels.length;i++) {
      double[] feature = features.get(labels[i]);
      if(null == feature) continue;
      for(int j=0;j<inputBands;j++) {
        filter.set(new int[]{0,0,i*inputBands+j},feature[j]);
      }
    }
    return this;
  }
  
  private void readFeatures() {
    if(null != filter && selected.length>0) {
      for(int i=0;i<selected.length;i++) {
        int offset = i * inputBands;
        double[] feature = new double[inputBands];
        for(int j=0;j<inputBands;j++) {
          feature[j] = filter.get(0,0, offset+j);
        }
        features.put(selected[i], feature);
      }
    }
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    CuDNN.setDevice(nncontext.getCudaDeviceId());
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.getDimensions();
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
          0, 0,1, 1, CUDNN_CONVOLUTION);
      final Pointer betaPtr = Pointer.to(new float[]{0.0f});
      final Pointer alphaPtr = Pointer.to(new float[]{1.0f});

      final float[] filterData = this.filter.getDataAsFloats();
      CudaPtr filterPtr = CuDNN.write(nncontext.getCudaDeviceId(), filterData);
      assert(0 < filterData.length);
      CudaPtr inputData = CudaPtr.toDeviceAsFloat(nncontext.getCudaDeviceId(), batch);
      assert kernelSize[0] * kernelSize[1] * kernelSize[2] == filterData.length;

      CudaPtr outputBuffer = CuDNN.alloc(nncontext.getCudaDeviceId(), Tensor.dim(outputSize) * 1l * length * Sizeof.FLOAT);
      CuDNN.devicePool.with(device -> {
        try {
          assert verifyOutputDims(inputDescriptor, filterDescriptor, convolutionDescriptor, outputSize);
          int algorithm = device.getForwardAlgorithm(
                  inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
          CudaPtr workSpace = device.allocateForwardWorkspace(nncontext.getCudaDeviceId(),
            inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
          CuDNN.handle(cudnnConvolutionForward(device.cudnnHandle, alphaPtr,
                  inputDescriptor.getPtr(), inputData.getPtr(),
                  filterDescriptor.getPtr(), filterPtr.getPtr(),
                  convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, betaPtr,
                  outputDescriptor.getPtr(), outputBuffer.getPtr()));
          workSpace.finalize();
        } catch (Throwable e) {
          throw new RuntimeException("Error map " + Arrays.toString(kernelSize),e);
        }
      });
      TensorList output = CudaPtr.fromDeviceFloat(outputBuffer, length, outputSize);

      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          outputBuffer.finalize();
          CuDNN.setDevice(nncontext.getCudaDeviceId());
          assert (error.length() == batch.length());
          int length = error.length();
          CudaPtr errorPtr = CudaPtr.toDeviceAsFloat(nncontext.getCudaDeviceId(), error);
          if (!isFrozen()) {
            CudaPtr filterBuffer = CuDNN.alloc(nncontext.getCudaDeviceId(), filterData.length * 1l * Sizeof.FLOAT);
            try {
              CuDNN.devicePool.with(device -> {
                int algorithm = device.getBackwardFilterAlgorithm(
                        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
                CudaPtr workSpace = device.allocateBackwardFilterWorkspace(nncontext.getCudaDeviceId(),
                  inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
                CuDNN.handle(cudnnConvolutionBackwardFilter(device.cudnnHandle,
                  alphaPtr, inputDescriptor.getPtr(), inputData.getPtr(),
                        outputDescriptor.getPtr(), errorPtr.getPtr(),
                        convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size,
                  betaPtr, filterDescriptor.getPtr(), filterBuffer.getPtr()));
                workSpace.finalize();
              });
            } catch (Throwable e) {
              throw new RuntimeException("Error map " + Arrays.toString(kernelSize),e);
            }
            final Tensor weightGradient = CudaPtr.fromDeviceFloat(filterBuffer, SchemaOutputLayer.this.filter.getDimensions());
            buffer.get(SchemaOutputLayer.this, SchemaOutputLayer.this.filter).accumulate(weightGradient.getData());
            filterBuffer.finalize();
          }
          if (input.isAlive()) {
            CudaPtr inputBuffer = CuDNN.alloc(nncontext.getCudaDeviceId(), Tensor.dim(batch.getDimensions()) * 1l * length * Sizeof.FLOAT);
            try {
              CuDNN.devicePool.with(device -> {
                int algorithm = device.getBackwardDataAlgorithm(
                        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
                CudaPtr workSpace = device.allocateBackwardDataWorkspace(nncontext.getCudaDeviceId(),
                  inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
                CuDNN.handle(cudnnConvolutionBackwardData(device.cudnnHandle, alphaPtr,
                        filterDescriptor.getPtr(), filterPtr.getPtr(),
                        outputDescriptor.getPtr(), errorPtr.getPtr(),
                        convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, betaPtr,
                        inputDescriptor.getPtr(), inputBuffer.getPtr()));
                workSpace.finalize();
              });
            } catch (Throwable e) {
              throw new RuntimeException("Error map " + Arrays.toString(kernelSize),e);
            }
            TensorList inputBufferTensors = CudaPtr.fromDeviceFloat(inputBuffer, length, inputSize);
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
      throw new RuntimeException("Error map image res " + Arrays.toString(inputSize),e);
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
      } else if(false) {
        x = inputSize[i] / (i==0? 1 : 1);
      } else {
        x = (1 + inputSize[i] - kernelSize[i]) / (i==0? 1 : 1);
      }
      if (0 >= x) {
        assert false;
      }
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
    if(4 != outputDims.length) {
      return false;
    }
    if(outputSize[0] != outputDims[3]) {
      return false;
    }
    if(outputSize[1] != outputDims[2]) {
      return false;
    }
    if(outputSize[2] != outputDims[1]) {
      return false;
    }
    return true;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.filter.getData());
  }
  
}
