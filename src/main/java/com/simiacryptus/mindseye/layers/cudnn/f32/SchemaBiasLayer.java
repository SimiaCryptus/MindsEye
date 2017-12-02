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
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.cudnnAddTensor;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardBias;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Schema bias layer.
 */
public class SchemaBiasLayer extends NNLayer implements SchemaComponent {
  
  private final Map<String, Double> features = new HashMap<>();
  private double[] bias;
  private String[] selected = {};
  
  /**
   * Instantiates a new Schema bias layer.
   *
   * @param json the json
   */
  protected SchemaBiasLayer(JsonObject json) {
    super(json);
    JsonObject featuresJson = json.getAsJsonObject("features");
    for (Map.Entry<String, JsonElement> e : featuresJson.entrySet()) {
      features.put(e.getKey(), e.getValue().getAsDouble());
    }
    JsonArray selectedJson = json.getAsJsonArray("selected");
    setSchema(IntStream.range(0, selectedJson.size()).mapToObj(i -> selectedJson.get(i).getAsString()).toArray(i -> new String[i]));
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
  }
  
  /**
   * Instantiates a new Schema bias layer.
   */
  public SchemaBiasLayer() {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
  }
  
  /**
   * From json schema bias layer.
   *
   * @param json the json
   * @return the schema bias layer
   */
  public static SchemaBiasLayer fromJson(JsonObject json) {
    return new SchemaBiasLayer(json);
  }
  
  public JsonObject getJson() {
    readFeatures();
    JsonObject json = super.getJsonStub();
    JsonArray jsonSelected = new JsonArray();
    for (String s : selected) jsonSelected.add(new JsonPrimitive(s));
    json.add("selected", jsonSelected);
    JsonObject jsonObject = new JsonObject();
    for (Map.Entry<String, Double> e : features.entrySet()) {
      jsonObject.add(e.getKey(), new JsonPrimitive(e.getValue()));
    }
    json.add("features", jsonObject);
    return json;
  }
  
  @Override
  public SchemaBiasLayer setSchema(String... labels) {
    if (null == labels) throw new IllegalArgumentException();
    readFeatures();
    selected = labels;
    bias = IntStream.range(0, labels.length).mapToDouble(i -> features.getOrDefault(labels[i], 0.0)).toArray();
    return this;
  }
  
  private void readFeatures() {
    if (null != bias && selected.length > 0) {
      for (int i = 0; i < selected.length; i++) {
        features.put(selected[i], bias[i]);
      }
    }
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    ((CudaExecutionContext) nncontext).initThread();
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.getDimensions();
    assert (inputSize[2] == bias.length);
    int[] outputSize = inputSize;
    int length = batch.length();
    
    try {
      
      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CudaResource<cudnnTensorDescriptor> filterDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, inputSize[2], 1, 1);
      
      assert (0 < this.bias.length);
      CudaPtr filterPtr = CuDNN.write(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.toFloats(this.bias));
      // Warning: For on-gpu operations, this modifies input mem buffer and can interfere with sibling consumers
      CudaPtr inputData = CudaPtr.toDeviceAsFloat(((CudaExecutionContext) nncontext).getDeviceNumber(), batch);
      try {
        CuDNN.handle(cudnnAddTensor(((CuDNN) nncontext).cudnnHandle,
          Pointer.to(new float[]{1.0f}),
          filterDescriptor.getPtr(), filterPtr.getPtr(),
          Pointer.to(new float[]{1.0f}),
          inputDescriptor.getPtr(), inputData.getPtr()));
      } catch (Throwable e) {
        throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
      }
      filterPtr.finalize();
      TensorList output = CudaPtr.fromDeviceFloat(inputData, length, outputSize, ((CuDNN) nncontext).cudnnHandle);
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
          ((CudaExecutionContext) nncontext).initThread();
          assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(Double::isFinite);
          CudaPtr errorPtr = CudaPtr.toDeviceAsFloat(((CudaExecutionContext) nncontext).getDeviceNumber(), error);
          if (!isFrozen()) {
            CudaPtr filterBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), SchemaBiasLayer.this.bias.length * 1l * Sizeof.FLOAT);
            try {
              CuDNN.handle(cudnnConvolutionBackwardBias(((CuDNN) nncontext).cudnnHandle,
                Pointer.to(new float[]{1.0f}),
                inputDescriptor.getPtr(), errorPtr.getPtr(),
                Pointer.to(new float[]{1.0f}),
                filterDescriptor.getPtr(), filterBuffer.getPtr()));
            } catch (Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(inputSize), e);
            }
            final Tensor weightGradient = CudaPtr.fromDeviceFloat(filterBuffer, new int[]{1, 1, inputSize[2]});
            //assert Arrays.stream(weightGradient.getData()).allMatch(Double::isFinite);
            Delta<NNLayer> deltaBuffer = buffer.get(SchemaBiasLayer.this, SchemaBiasLayer.this.bias);
            deltaBuffer.addInPlace(weightGradient.getData());
            //assert Arrays.stream(deltaBuffer.delta).allMatch(Double::isFinite);
            filterBuffer.finalize();
          }
          if (input.isAlive()) {
            input.accumulate(buffer, error);
          }
        }
        
        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
    }
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.getBias());
  }
  
  /**
   * Get bias double [ ].
   *
   * @return the double [ ]
   */
  public double[] getBias() {
    //assert Arrays.stream(this.bias).allMatch(Double::isFinite);
    return bias;
  }
}
