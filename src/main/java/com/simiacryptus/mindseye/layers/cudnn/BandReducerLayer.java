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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.GpuSystem;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer.PoolingMode;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Similar to the pooling layer, but the pool size is always the image size. The output dimensions are always 1x1xN.
 */
@SuppressWarnings("serial")
public class BandReducerLayer extends NNLayer implements MultiPrecision<BandReducerLayer> {
  
  private PoolingLayer.PoolingMode mode = PoolingLayer.PoolingMode.Max;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Pooling layer.
   */
  public BandReducerLayer() {
    super();
  }
  
  /**
   * Instantiates a new Pooling layer.
   *
   * @param json the json
   */
  protected BandReducerLayer(final @NotNull JsonObject json) {
    super(json);
    mode = Arrays.stream(PoolingLayer.PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt()).findFirst().get();
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * From json pooling layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the pooling layer
   */
  public static BandReducerLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new BandReducerLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public @NotNull NNLayer getCompatibilityLayer() {
    throw new RuntimeException("Not Implemented");
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    if (!GpuSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.getDimensions();
    @NotNull PoolingLayer impl = new PoolingLayer().setMode(mode).setPrecision(precision)
                                                   .setWindowX(inputSize[1])
                                                   .setWindowY(inputSize[0]);
    NNResult result = impl.eval(inObj);
    impl.freeRef();
    return result;
  }
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final @NotNull JsonObject json = super.getJsonStub();
    json.addProperty("mode", mode.id);
    json.addProperty("precision", precision.name());
    return json;
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
  public @NotNull BandReducerLayer setMode(final PoolingMode mode) {
    this.mode = mode;
    return this;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public @NotNull BandReducerLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Override
  public @NotNull List<double[]> state() {
    return Arrays.asList();
  }
  
}
