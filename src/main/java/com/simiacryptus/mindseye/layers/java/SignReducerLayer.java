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
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.UUID;

/**
 * The type Sign reducer layer.
 */
@SuppressWarnings("serial")
public class SignReducerLayer extends DAGNetwork {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SignReducerLayer.class);
  private final DAGNode head;
  
  /**
   * Instantiates a new Sign reducer layer.
   */
  public SignReducerLayer() {
    super(1);
    final DAGNode avgInput = add(new AvgReducerLayer(), getInput(0));
    final DAGNode stdDevInput = add(new NthPowerActivationLayer().setPower(0.5),
                                    add(new SumInputsLayer(),
                                        add(new AvgReducerLayer(), add(new SqActivationLayer(), getInput(0))),
                                        add(new LinearActivationLayer().setScale(-1), add(new SqActivationLayer(), avgInput))
                                       ));
    head = add(new SigmoidActivationLayer().setBalanced(false),
               add(new ProductInputsLayer(),
                   avgInput,
                   add(new NthPowerActivationLayer().setPower(-1), stdDevInput)));
  }
  
  /**
   * Instantiates a new Sign reducer layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected SignReducerLayer(final JsonObject json, Map<String, byte[]> rs) {
    super(json, rs);
    head = nodesById.get(UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
  }
  
  /**
   * From json nn layer.
   *
   * @param inner the inner
   * @param rs    the rs
   * @return the nn layer
   */
  public static NNLayer fromJson(final JsonObject inner, Map<String, byte[]> rs) {
    return new SignReducerLayer(inner, rs);
  }
  
  @Override
  public DAGNode getHead() {
    return head;
  }
}
