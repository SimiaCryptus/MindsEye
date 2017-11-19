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
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.UUID;

/**
 * The type Target value layer.
 */
@SuppressWarnings("serial")
public class TargetValueLayer extends DAGNetwork {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TargetValueLayer.class);
  private final DAGNode head;
  private final DAGNode target;
  
  /**
   * Instantiates a new Target value layer.
   *
   * @param json the json
   */
  protected TargetValueLayer(JsonObject json) {
    super(json);
    head = nodesById.get(UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
    target = nodesById.get(UUID.fromString(json.getAsJsonPrimitive("target").getAsString()));
  }
  
  /**
   * Instantiates a new Target value layer.
   *
   * @param values the values
   */
  public TargetValueLayer(double... values) {
    super(1);
    this.target = add(new ConstNNLayer(new Tensor(values)));
    this.head = add(new MeanSqLossLayer(), getInput(0), target);
  }
  
  /**
   * From json nn layer.
   *
   * @param inner the inner
   * @return the nn layer
   */
  public static NNLayer fromJson(JsonObject inner) {
    return new TargetValueLayer(inner);
  }
  
  @Override
  public JsonObject getJson() {
    JsonObject json = super.getJson();
    json.addProperty("target", target.toString());
    return json;
  }
  
  @Override
  public DAGNode getHead() {
    return head;
  }
  
  /**
   * Sets target.
   *
   * @param value the value
   * @return the target
   */
  public TargetValueLayer setTarget(double... value) {
    target.<ConstNNLayer>getLayer().setData(new Tensor(value));
    return this;
  }
}
