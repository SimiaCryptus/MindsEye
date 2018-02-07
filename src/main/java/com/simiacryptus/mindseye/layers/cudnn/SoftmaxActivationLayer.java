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
import com.simiacryptus.mindseye.lang.cudnn.GpuSystem;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The classic "softmax" layer. All outputs will sum to 1 and be proportional to the log of the input.
 */
@SuppressWarnings("serial")
public class SoftmaxActivationLayer extends NNLayer implements MultiPrecision<SoftmaxActivationLayer> {
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);
  
  private Precision precision = Precision.Double;
  
  
  /**
   * Instantiates a new Activation layer.
   */
  public SoftmaxActivationLayer() {
  
  }
  
  /**
   * Instantiates a new Activation layer.
   *
   * @param json the json
   */
  protected SoftmaxActivationLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * From json activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the activation layer
   */
  public static SoftmaxActivationLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new SoftmaxActivationLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @javax.annotation.Nonnull
  public NNLayer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer.class);
  }
  
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    if (!GpuSystem.isEnabled()) return getCompatibilityLayer().eval(inObj);
    log.debug("Not Implemented: " + getClass().getCanonicalName());
    return getCompatibilityLayer().eval(inObj);
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public SoftmaxActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
