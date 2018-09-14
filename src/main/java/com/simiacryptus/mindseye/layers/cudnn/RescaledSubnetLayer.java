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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * This layer works as a scaling function, similar to a father wavelet. Allows convolutional and pooling layers to work
 * across larger png regions. Implemented via CudaSystem.
 */
@SuppressWarnings("serial")
public class RescaledSubnetLayer extends LayerBase implements MultiPrecision<RescaledSubnetLayer> {
  private static final Logger log = LoggerFactory.getLogger(RescaledSubnetLayer.class);
  
  private int scale;
  private Layer layer;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img eval layer.
   */
  private RescaledSubnetLayer() {
  }
  
  /**
   * Instantiates a new Rescaled subnet layer.
   *
   * @param scale the scale
   * @param layer the layer
   */
  public RescaledSubnetLayer(int scale, Layer layer) {
    this.scale = scale;
    this.layer = layer;
  }
  
  /**
   * Instantiates a new Img eval layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected RescaledSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    scale = json.get("scale").getAsInt();
    layer = Layer.fromJson(json, rs);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json img eval layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img eval layer
   */
  public static RescaledSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new RescaledSubnetLayer(json, rs);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.RescaledSubnetLayer(scale, layer);
  }
  
  @Nullable
  @Override
  public Result evalAndFree(final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    log.warn("Not Implemented: " + getClass().getCanonicalName());
    return getCompatibilityLayer().evalAndFree(inObj);
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("scale", scale);
    json.add("layer", layer.getJson(resources, dataSerializer));
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public RescaledSubnetLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
}
