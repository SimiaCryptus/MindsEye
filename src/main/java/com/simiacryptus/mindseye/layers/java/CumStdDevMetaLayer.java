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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * The type Std dev meta layer.
 */
@SuppressWarnings("serial")
public class CumStdDevMetaLayer extends DAGNetwork implements CumSum {
  
  /**
   * From json nn layer.
   *
   * @param inner the inner
   * @return the nn layer
   */
  public static NNLayer fromJson(JsonObject inner) {
    return new CumStdDevMetaLayer(inner);
  }
  
  /**
   * Instantiates a new Std dev meta layer.
   *
   * @param json the json
   */
  protected CumStdDevMetaLayer(JsonObject json) {
    super(json);
    head = nodesById.get(UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
    JsonArray children = json.getAsJsonArray("children");
    for(int i=0;i<children.size();i++) {
      cumsumChildren.add((CumSum) layersById.get(UUID.fromString(children.get(i).getAsString())));
    }
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CumStdDevMetaLayer.class);
  private final DAGNode head;
  
  /**
   * Instantiates a new Std dev meta layer.
   */
  public CumStdDevMetaLayer() {
    super(1);
    DAGNode input = getInput(0);
    DAGNode l0 = add(add(new CumSumMetaLayer()), add(new NthPowerActivationLayer().setPower(0), input));
    DAGNode l1 = add(add(new CumSumMetaLayer()), add(new NthPowerActivationLayer().setPower(1), input));
    DAGNode l2 = add(add(new CumSumMetaLayer()), add(new NthPowerActivationLayer().setPower(2), input));
    DAGNode l0inv = add(new NthPowerActivationLayer().setPower(-1), l0);
    DAGNode e1 = add(new ProductInputsLayer(), l1, l0inv);
    DAGNode e2 = add(new ProductInputsLayer(), l2, l0inv);
    this.head = add(new NthPowerActivationLayer().setPower(0.5),
      add(new SumInputsLayer(),
        e2,
        add(new LinearActivationLayer().setScale(-1),
          add(new NthPowerActivationLayer().setPower(2), e1))
      ));
  }
  
  @Override
  public DAGNode getHead() {
    return head;
  }
  
  
  private final List<CumSum> cumsumChildren = new ArrayList<>();
  
  private <T extends CumSum> T add(T obj) {
    cumsumChildren.add(obj);
    return obj;
  }
  
  @Override
  public JsonObject getJson() {
    JsonObject json = super.getJson();
    JsonArray childrenArray = new JsonArray();
    for (CumSum item : cumsumChildren) {
      childrenArray.add(new JsonPrimitive(((NNLayer) (item)).getId()));
    }
    json.add("children", childrenArray);
    return json;
  }
  
  @Override
  public double getCarryOver() {
    return cumsumChildren.get(0).getCarryOver();
  }
  
  @Override
  public CumStdDevMetaLayer setCarryOver(double carryOver) {
    cumsumChildren.forEach(x->x.setCarryOver(carryOver));
    return this;
  }
  
  @Override
  public int getCarryoverDenominator() {
    return cumsumChildren.get(0).getCarryoverDenominator();
  }
  
  @Override
  public CumStdDevMetaLayer setCarryoverDenominator(int carryoverDenominator) {
    cumsumChildren.forEach(x->x.setCarryoverDenominator(carryoverDenominator));
    return this;
  }
}
