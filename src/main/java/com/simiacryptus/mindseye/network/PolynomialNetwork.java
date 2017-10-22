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
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.activation.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
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
      return PolynomialNetwork.this.add(newNthPowerLayer(power), PolynomialNetwork.this.add(bias, PolynomialNetwork.this.add(factor, input)));
    }
  
    public JsonObject getJson() {
      JsonObject json = new JsonObject();
      json.addProperty("bias", bias.getId().toString());
      json.addProperty("factor", factor.getId().toString());
      json.addProperty("power", power);
      return json;
    }
  }
  
  protected NNLayer alpha = null;
  protected NNLayer alphaBias = null;
  protected DAGNode head;
  protected List<Correcton> corrections = new ArrayList<>();
  protected final int[] inputDims;
  protected final int[] outputDims;
  
  
  public JsonObject getJson() {
    assertConsistent();
    DAGNode head = getHead();
    JsonObject json = super.getJson();
    json.addProperty("head", head.getId().toString());
    if(null != alpha) json.addProperty("alpha", alpha.getId().toString());
    if(null != alphaBias) json.addProperty("alphaBias", alpha.getId().toString());
    json.add("inputDims", toJson(inputDims));
    json.add("outputDims", toJson(outputDims));
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
    if(json.get("alpha") != null) alpha = layersById.get(UUID.fromString(json.get("alpha").getAsString()));
    if(json.get("alphaBias") != null) alphaBias = layersById.get(UUID.fromString(json.get("alphaBias").getAsString()));
    inputDims = toIntArray(json.getAsJsonArray("inputDims"));
    outputDims = toIntArray(json.getAsJsonArray("outputDims"));
    json.getAsJsonArray("corrections").forEach(item->{
      corrections.add(new Correcton(item.getAsJsonObject()));
    });
  }
  
  public static JsonArray toJson(int[] dims) {
    JsonArray array = new JsonArray();
    for(int i:dims) array.add(new JsonPrimitive(i));
    return array;
  }
  
  public static int[] toIntArray(JsonArray dims) {
    int[] x = new int[dims.size()];
    int j = 0;
    for(Iterator<JsonElement> i = dims.iterator(); i.hasNext();) {
      x[j++] = i.next().getAsInt();
    }
    return x;
  }
  
  public PolynomialNetwork(int[] inputDims,int[] outputDims) {
    super(1);
    this.inputDims = inputDims;
    this.outputDims = outputDims;
  }
  
  public NNLayer newBias(int[] dims, double weight) {
    return new BiasLayer(dims).setWeights(i->weight);
  }
  
  public NNLayer newSynapse(double weight) {
    return new DenseSynapseLayer(inputDims, outputDims).setWeights(() -> weight * (Math.random() - 1));
  }
  
  public synchronized DAGNode getHead() {
    if(null == head) {
      synchronized (this) {
        if(null == head) {
          if(null == alpha) {
            this.alpha = newSynapse(1e-8);
            this.alphaBias = newBias(inputDims, 0.0);
          }
          this.reset();
          DAGNode input = getInput(0);
          ArrayList<DAGNode> terms = new ArrayList<>();
          terms.add(add(alpha,add(alphaBias,input)));
          for(Correcton c : corrections) {
            terms.add(c.add(input));
          }
          head = terms.size()==1?terms.get(0):add(newProductLayer(), terms.toArray(new DAGNode[]{}));
        }
      }
    }
    return head;
  }
  
  public NNLayer newNthPowerLayer(double power) {
    return new NthPowerActivationLayer().setPower(power);
  }
  
  public NNLayer newProductLayer() {
    return new com.simiacryptus.mindseye.layers.reducers.ProductInputsLayer();
  }
  
  public void addTerm(double power) {
    corrections.add(new Correcton(power,
      newBias(outputDims, 1.0),
      newSynapse(0.0)
    ));
  }
  
}
