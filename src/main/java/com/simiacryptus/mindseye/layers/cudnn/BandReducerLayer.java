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
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer.PoolingMode;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Similar to the pooling layer, but the pool size is always the png size. The output dimensions are always 1x1xN.
 */
@SuppressWarnings("serial")
public class BandReducerLayer extends LayerBase implements MultiPrecision<BandReducerLayer> {

  private PoolingLayer.PoolingMode mode = PoolingLayer.PoolingMode.Max;
  private Precision precision = Precision.Double;
  private double alpha = 1.0;

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
  protected BandReducerLayer(@Nonnull final JsonObject json) {
    super(json);
    mode = Arrays.stream(PoolingLayer.PoolingMode.values()).filter(i -> i.id == json.get("mode").getAsInt()).findFirst().get();
    precision = Precision.valueOf(json.get("precision").getAsString());
    alpha = json.get("alpha").getAsDouble();
  }

  /**
   * From json pooling layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the pooling layer
   */
  public static BandReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BandReducerLayer(json);
  }

  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    throw new RuntimeException("Not Implemented");
  }

  @Nullable
  @Override
  public Result evalAndFree(final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    @Nonnull final int[] inputSize = batch.getDimensions();
    @Nonnull PoolingLayer impl = new PoolingLayer().setMode(mode).setPrecision(precision)
        .setWindowX(inputSize[1])
        .setWindowY(inputSize[0])
        .setAlpha(alpha);
    @Nullable Result result = impl.evalAndFree(inObj);
    impl.freeRef();
    return result;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("alpha", alpha);
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
  @Nonnull
  public BandReducerLayer setMode(final PoolingMode mode) {
    this.mode = mode;
    return this;
  }

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Nonnull
  @Override
  public BandReducerLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

  /**
   * Gets alphaList.
   *
   * @return the alphaList
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Sets alphaList.
   *
   * @param alpha the alphaList
   * @return the alphaList
   */
  public BandReducerLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }
}
