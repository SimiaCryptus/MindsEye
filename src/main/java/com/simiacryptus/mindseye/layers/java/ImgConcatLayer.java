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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Concatenates two or more images with the same resolution so the output contains all input color bands.
 */
@SuppressWarnings("serial")
public class ImgConcatLayer extends LayerBase {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgConcatLayer.class);
  private int maxBands;
  
  /**
   * Instantiates a new Img concat layer.
   */
  public ImgConcatLayer() {
    setMaxBands(0);
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   */
  protected ImgConcatLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    JsonElement maxBands = json.get("maxBands");
    if (null != maxBands) setMaxBands(maxBands.getAsInt());
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
  
  @Nullable
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    assert Arrays.stream(inObj).allMatch(x -> x.getData().getDimensions().length == 3) : "This component is for use mapCoords 3d image tensors only";
    final int numBatches = inObj[0].getData().length();
    assert Arrays.stream(inObj).allMatch(x -> x.getData().length() == numBatches) : "All inputs must use same batch size";
    @javax.annotation.Nonnull final int[] outputDims = Arrays.copyOf(inObj[0].getData().getDimensions(), 3);
    outputDims[2] = Arrays.stream(inObj).mapToInt(x -> x.getData().getDimensions()[2]).sum();
    if (maxBands > 0) outputDims[2] = Math.min(maxBands, outputDims[2]);
    assert Arrays.stream(inObj).allMatch(x -> x.getData().getDimensions()[0] == outputDims[0]) : "Inputs must be same size";
    assert Arrays.stream(inObj).allMatch(x -> x.getData().getDimensions()[1] == outputDims[1]) : "Inputs must be same size";
    
    @javax.annotation.Nonnull final List<Tensor> outputTensors = new ArrayList<>();
    for (int b = 0; b < numBatches; b++) {
      @javax.annotation.Nonnull final Tensor outputTensor = new Tensor(outputDims);
      int pos = 0;
      @Nullable final double[] outputTensorData = outputTensor.getData();
      for (int i = 0; i < inObj.length; i++) {
        @javax.annotation.Nullable Tensor tensor = inObj[i].getData().get(b);
        @Nullable final double[] data = tensor.getData();
        System.arraycopy(data, 0, outputTensorData, pos, Math.min(data.length, outputTensorData.length - pos));
        pos += data.length;
        tensor.freeRef();
      }
      outputTensors.add(outputTensor);
    }
    return new Result(TensorArray.wrap(outputTensors.toArray(new Tensor[]{})), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      assert numBatches == data.length();
      
      @javax.annotation.Nonnull final List<Tensor[]> splitBatches = new ArrayList<>();
      for (int b = 0; b < numBatches; b++) {
        @javax.annotation.Nullable final Tensor tensor = data.get(b);
        @javax.annotation.Nonnull final Tensor[] outputTensors2 = new Tensor[inObj.length];
        int pos = 0;
        for (int i = 0; i < inObj.length; i++) {
          @javax.annotation.Nonnull final Tensor dest = new Tensor(inObj[i].getData().getDimensions());
          @Nullable double[] tensorData = tensor.getData();
          System.arraycopy(tensorData, pos, dest.getData(), 0, Math.min(dest.size(), tensorData.length - pos));
          pos += dest.size();
          outputTensors2[i] = dest;
        }
        tensor.freeRef();
        splitBatches.add(outputTensors2);
      }
      
      @javax.annotation.Nonnull final Tensor[][] splitData = new Tensor[inObj.length][];
      for (int i = 0; i < splitData.length; i++) {
        splitData[i] = new Tensor[numBatches];
      }
      for (int i = 0; i < inObj.length; i++) {
        for (int b = 0; b < numBatches; b++) {
          splitData[i][b] = splitBatches.get(b)[i];
        }
      }
      
      for (int i = 0; i < inObj.length; i++) {
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(splitData[i]);
        inObj[i].accumulate(buffer, tensorArray);
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        for (@javax.annotation.Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
      
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull JsonObject json = super.getJsonStub();
    json.addProperty("maxBands", maxBands);
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
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
  public ImgConcatLayer setMaxBands(int maxBands) {
    this.maxBands = maxBands;
    return this;
  }
}
