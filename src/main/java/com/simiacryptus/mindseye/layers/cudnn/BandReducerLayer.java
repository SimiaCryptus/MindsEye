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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.NNExecutionContext;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer.PoolingMode;

import java.util.Arrays;
import java.util.List;

/**
 * The type Pooling layer.
 */
public class BandReducerLayer extends NNLayer implements LayerPrecision<BandReducerLayer> {
  
  private PoolingLayer.PoolingMode mode = PoolingLayer.PoolingMode.Max;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Pooling layer.
   *
   * @param json the json
   */
  protected BandReducerLayer(JsonObject json) {
    super(json);
    mode = Arrays.stream(PoolingLayer.PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt()).findFirst().get();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * Instantiates a new Pooling layer.
   */
  public BandReducerLayer() {
    super();
  }
  
  /**
   * From json pooling layer.
   *
   * @param json the json
   * @return the pooling layer
   */
  public static BandReducerLayer fromJson(JsonObject json) {
    return new BandReducerLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode.id);
    json.addProperty("precision",precision.name());
    return json;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.getDimensions();
    return new PoolingLayer().setMode(mode).setPrecision(precision).setWindowX(inputSize[0]).setWindowY(inputSize[1]).eval(nncontext, inObj);
  }
  
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  /**
   * Gets mode.
   *
   * @return the mode
   */
  public PoolingMode getMode() {
    return mode;
  }
  
  /**
   * Sets mode.
   *
   * @param mode the mode
   * @return the mode
   */
  public BandReducerLayer setMode(PoolingMode mode) {
    this.mode = mode;
    return this;
  }
  
  public Precision getPrecision() {
    return precision;
  }
  
  public BandReducerLayer setPrecision(Precision precision) {
    this.precision = precision;
    return this;
  }
  
}
