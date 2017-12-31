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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.NNExecutionContext;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The classic "softmax" layer. All outputs will sum to 1 and be proportional to the log of the input.
 */
@SuppressWarnings("serial")
public class SoftmaxActivationLayer extends NNLayer implements LayerPrecision<SoftmaxActivationLayer> {
  private static final Logger logger = LoggerFactory.getLogger(SoftmaxActivationLayer.class);
  
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
  protected SoftmaxActivationLayer(final JsonObject json) {
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
  public static SoftmaxActivationLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new SoftmaxActivationLayer(json);
  }
  
  public NNLayer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer();
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    if (((CudaExecutionContext) nncontext).getDeviceNumber() < 0) return getCompatibilityLayer().eval(nncontext, inObj);
    
    logger.warn("Not Implemented: " + getClass().getCanonicalName());
    return getCompatibilityLayer().eval(nncontext, inObj);
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public SoftmaxActivationLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
