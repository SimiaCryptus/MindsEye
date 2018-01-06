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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Concatenates two or more images with the same resolution so the output contains all input color bands.
 */
@SuppressWarnings("serial")
public class ImgConcatLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgConcatLayer.class);
  
  /**
   * Instantiates a new Img concat layer.
   */
  public ImgConcatLayer() {
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param id the id
   */
  protected ImgConcatLayer(final JsonObject id) {
    super(id);
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
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    
    assert Arrays.stream(inObj).allMatch(x -> x.getData().get(0).getDimensions().length == 3) : "This component is for use mapCoords 3d image tensors only";
    final int numBatches = inObj[0].getData().length();
    assert Arrays.stream(inObj).allMatch(x -> x.getData().length() == numBatches) : "All inputs must use same batch size";
    final int[] outputDims = Arrays.copyOf(inObj[0].getData().get(0).getDimensions(), 3);
    outputDims[2] = Arrays.stream(inObj).mapToInt(x -> x.getData().get(0).getDimensions()[2]).sum();
    assert Arrays.stream(inObj).allMatch(x -> x.getData().get(0).getDimensions()[0] == outputDims[0]) : "Inputs must be same size";
    assert Arrays.stream(inObj).allMatch(x -> x.getData().get(0).getDimensions()[1] == outputDims[1]) : "Inputs must be same size";
  
    final List<Tensor> outputTensors = new ArrayList<>();
    for (int b = 0; b < numBatches; b++) {
      final Tensor outputTensor = new Tensor(outputDims);
      int pos = 0;
      final double[] outputTensorData = outputTensor.getData();
      for (int i = 0; i < inObj.length; i++) {
        final double[] data = inObj[i].getData().get(b).getData();
        System.arraycopy(data, 0, outputTensorData, pos, data.length);
        pos += data.length;
      }
      outputTensors.add(outputTensor);
    }
    return new NNResult(outputTensors.toArray(new Tensor[]{})) {
  
      @Override
      public void free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        assert numBatches == data.length();
    
        final List<Tensor[]> splitBatches = new ArrayList<>();
        for (int b = 0; b < numBatches; b++) {
          final Tensor tensor = data.get(b);
          final Tensor[] outputTensors = new Tensor[inObj.length];
          int pos = 0;
          for (int i = 0; i < inObj.length; i++) {
            final Tensor dest = new Tensor(inObj[i].getData().get(0).getDimensions());
            System.arraycopy(tensor.getData(), pos, dest.getData(), 0, dest.size());
            pos += dest.size();
            outputTensors[i] = dest;
          }
          splitBatches.add(outputTensors);
        }
    
        final Tensor[][] splitData = new Tensor[inObj.length][];
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
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
