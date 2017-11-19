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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The type Img concat layer.
 */
public class ImgConcatLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @return the img concat layer
   */
  public static ImgConcatLayer fromJson(JsonObject json) {
    return new ImgConcatLayer(json);
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param id the id
   */
  protected ImgConcatLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgConcatLayer.class);
  
  /**
   * Instantiates a new Img concat layer.
   */
  public ImgConcatLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    
    assert Arrays.stream(inObj).allMatch(x -> x.getData().get(0).getDimensions().length == 3) : "This component is for use mapCoords 3d image tensors only";
    int numBatches = inObj[0].getData().length();
    assert Arrays.stream(inObj).allMatch(x -> x.getData().length() == numBatches) : "All inputs must use same batch size";
    int[] outputDims = Arrays.copyOf(inObj[0].getData().get(0).getDimensions(), 3);
    outputDims[2] = Arrays.stream(inObj).mapToInt(x -> x.getData().get(0).getDimensions()[2]).sum();
    assert Arrays.stream(inObj).allMatch(x -> x.getData().get(0).getDimensions()[0] == outputDims[0]) : "Inputs must be same size";
    assert Arrays.stream(inObj).allMatch(x -> x.getData().get(0).getDimensions()[1] == outputDims[1]) : "Inputs must be same size";
    
    List<Tensor> outputTensors = new ArrayList<>();
    for (int b = 0; b < numBatches; b++) {
      Tensor outputTensor = new Tensor(outputDims);
      int pos = 0;
      double[] outputTensorData = outputTensor.getData();
      for (int i = 0; i < inObj.length; i++) {
        double[] data = inObj[i].getData().get(b).getData();
        System.arraycopy(data, 0, outputTensorData, pos, data.length);
        pos += data.length;
      }
      outputTensors.add(outputTensor);
    }
    return new NNResult(outputTensors.toArray(new Tensor[]{})) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        assert (numBatches == data.length());
        
        List<Tensor[]> splitBatches = new ArrayList<>();
        for (int b = 0; b < numBatches; b++) {
          Tensor tensor = data.get(b);
          Tensor[] outputTensors = new Tensor[inObj.length];
          int pos = 0;
          for (int i = 0; i < inObj.length; i++) {
            Tensor dest = new Tensor(inObj[i].getData().get(0).getDimensions());
            System.arraycopy(tensor.getData(), pos, dest.getData(), 0, dest.size());
            pos += dest.size();
            outputTensors[i] = dest;
          }
          splitBatches.add(outputTensors);
        }
        
        Tensor[][] splitData = new Tensor[inObj.length][];
        for (int i = 0; i < splitData.length; i++) {
          splitData[i] = new Tensor[numBatches];
        }
        for (int i = 0; i < inObj.length; i++) {
          for (int b = 0; b < numBatches; b++) {
            splitData[i][b] = splitBatches.get(b)[i];
          }
        }
        
        for (int i = 0; i < inObj.length; i++) {
          inObj[i].accumulate(buffer, new TensorArray(splitData[i]));
        }
      }
      
      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
