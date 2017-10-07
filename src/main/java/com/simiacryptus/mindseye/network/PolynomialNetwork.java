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

package com.simiacryptus.mindseye.network;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.activation.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.reducers.ProductInputsLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;

import java.util.ArrayList;
import java.util.List;
import java.util.Spliterators;
import java.util.UUID;

public class PolynomialNetwork extends DAGNetwork {
  
  public class Correcton {
    public final double power;
    public final NNLayer bias;
    public final NNLayer factor;
  
    public Correcton(double power, NNLayer bias, NNLayer factor) {
      this.power = power;
      this.bias = bias;
      this.factor = factor;
    }
  
    public Correcton(JsonObject json) {
      this.power = json.get("power").getAsDouble();
      this.bias = layersById.get(UUID.fromString(json.get("bias").getAsString()));
      this.factor = layersById.get(UUID.fromString(json.get("factor").getAsString()));
    }
  
    public DAGNode add(DAGNode input) {
      return PolynomialNetwork.this.add(new NthPowerActivationLayer().setPower(power), PolynomialNetwork.this.add(bias, PolynomialNetwork.this.add(factor, input)));
    }
    public JsonObject getJson() {
      JsonObject json = new JsonObject();
      json.addProperty("bias", bias.getId().toString());
      json.addProperty("factor", factor.getId().toString());
      json.addProperty("power", power);
      return json;
    }
  }
  
  private DenseSynapseLayer alpha = null;
  private BiasLayer alphaBias = null;
  private DAGNode head;
  private List<Correcton> corrections = new ArrayList<>();
  
  public JsonObject getJson() {
    assertConsistent();
    DAGNode head = getHead();
    JsonObject json = super.getJson();
    json.addProperty("head", head.getId().toString());
    if(null != alpha) json.addProperty("alpha", alpha.getId().toString());
    if(null != alphaBias) json.addProperty("alphaBias", alpha.getId().toString());
    JsonArray elements = new JsonArray();
    for(Correcton c : corrections) elements.add(c.getJson());
    json.add("corrections", elements);
    assert null != NNLayer.fromJson(json) : "Smoke test deserialization";
    return json;
  }
  
  public static PolynomialNetwork fromJson(JsonObject json) {
    return new PolynomialNetwork(json);
  }
  
  protected PolynomialNetwork(JsonObject json) {
    super(json);
    head = nodesById.get(UUID.fromString(json.get("head").getAsString()));
    if(json.get("alpha") != null) alpha = (DenseSynapseLayer) layersById.get(UUID.fromString(json.get("alpha").getAsString()));
    if(json.get("alphaBias") != null) alphaBias = (BiasLayer) layersById.get(UUID.fromString(json.get("alphaBias").getAsString()));
    json.getAsJsonArray("corrections").forEach(item->{
      corrections.add(new Correcton(item.getAsJsonObject()));
    });
  }
  
  public PolynomialNetwork(DenseSynapseLayer alpha) {
    super(1);
    this.alpha = alpha;
    this.alphaBias = new BiasLayer(alpha.inputDims);
  }
  
  public synchronized DAGNode getHead() {
    if(null == head) {
      synchronized (this) {
        if(null == head) {
          this.reset();
          DAGNode input = getInput(0);
          ArrayList<DAGNode> terms = new ArrayList<>();
          terms.add(add(alpha,add(alphaBias,input)));
          for(Correcton c : corrections) {
            terms.add(c.add(input));
          }
          head = terms.size()==1?terms.get(0):add(new ProductInputsLayer(), terms.toArray(new DAGNode[]{}));
        }
      }
    }
    return head;
  }
  
  public void addTerm(double power) {
    corrections.add(new Correcton(power,
      new BiasLayer(alpha.outputDims).setWeights(i->1.0),
      new DenseSynapseLayer(alpha.inputDims, alpha.outputDims).setWeights(()->0.0)
    ));
  }
  
}
