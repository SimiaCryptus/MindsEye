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

package com.simiacryptus.mindseye.network.util;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.layers.java.ProductInputsLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.UUID;

/**
 * The type Polynomial network.
 */
@SuppressWarnings("serial")
public class PolynomialNetwork extends DAGNetwork {
  
  /**
   * The Input dims.
   */
  protected final int[] inputDims;
  /**
   * The Output dims.
   */
  protected final int[] outputDims;
  /**
   * The Alpha.
   */
  protected NNLayer alpha = null;
  /**
   * The Alpha bias.
   */
  protected NNLayer alphaBias = null;
  /**
   * The Corrections.
   */
  protected List<Correcton> corrections = new ArrayList<>();
  /**
   * The Head.
   */
  protected DAGNode head;
  
  /**
   * Instantiates a new Polynomial network.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public PolynomialNetwork(final int[] inputDims, final int[] outputDims) {
    super(1);
    this.inputDims = inputDims;
    this.outputDims = outputDims;
  }
  
  
  /**
   * Instantiates a new Polynomial network.
   *
   * @param json the json
   */
  protected PolynomialNetwork(final JsonObject json) {
    super(json);
    head = nodesById.get(UUID.fromString(json.get("head").getAsString()));
    if (json.get("alpha") != null) {
      alpha = layersById.get(UUID.fromString(json.get("alpha").getAsString()));
    }
    if (json.get("alphaBias") != null) {
      alphaBias = layersById.get(UUID.fromString(json.get("alphaBias").getAsString()));
    }
    inputDims = PolynomialNetwork.toIntArray(json.getAsJsonArray("inputDims"));
    outputDims = PolynomialNetwork.toIntArray(json.getAsJsonArray("outputDims"));
    json.getAsJsonArray("corrections").forEach(item -> {
      corrections.add(new Correcton(item.getAsJsonObject()));
    });
  }
  
  /**
   * From json polynomial network.
   *
   * @param json the json
   * @return the polynomial network
   */
  public static PolynomialNetwork fromJson(final JsonObject json) {
    return new PolynomialNetwork(json);
  }
  
  /**
   * To int array int [ ].
   *
   * @param dims the dims
   * @return the int [ ]
   */
  public static int[] toIntArray(final JsonArray dims) {
    final int[] x = new int[dims.size()];
    int j = 0;
    for (final Iterator<JsonElement> i = dims.iterator(); i.hasNext(); ) {
      x[j++] = i.next().getAsInt();
    }
    return x;
  }
  
  /**
   * To json json array.
   *
   * @param dims the dims
   * @return the json array
   */
  public static JsonArray toJson(final int[] dims) {
    final JsonArray array = new JsonArray();
    for (final int i : dims) {
      array.add(new JsonPrimitive(i));
    }
    return array;
  }
  
  /**
   * Add term.
   *
   * @param power the power
   */
  public void addTerm(final double power) {
    corrections.add(new Correcton(power,
      newBias(outputDims, 1.0),
      newSynapse(0.0)
    ));
  }
  
  @Override
  public synchronized DAGNode getHead() {
    if (null == head) {
      synchronized (this) {
        if (null == head) {
          if (null == alpha) {
            alpha = newSynapse(1e-8);
            alphaBias = newBias(inputDims, 0.0);
          }
          reset();
          final DAGNode input = getInput(0);
          final ArrayList<DAGNode> terms = new ArrayList<>();
          terms.add(add(alpha, add(alphaBias, input)));
          for (final Correcton c : corrections) {
            terms.add(c.add(input));
          }
          head = terms.size() == 1 ? terms.get(0) : add(newProductLayer(), terms.toArray(new DAGNode[]{}));
        }
      }
    }
    return head;
  }
  
  @Override
  public JsonObject getJson() {
    assertConsistent();
    final DAGNode head = getHead();
    final JsonObject json = super.getJson();
    json.addProperty("head", head.getId().toString());
    if (null != alpha) {
      json.addProperty("alpha", alpha.getId().toString());
    }
    if (null != alphaBias) {
      json.addProperty("alphaBias", alpha.getId().toString());
    }
    json.add("inputDims", PolynomialNetwork.toJson(inputDims));
    json.add("outputDims", PolynomialNetwork.toJson(outputDims));
    final JsonArray elements = new JsonArray();
    for (final Correcton c : corrections) {
      elements.add(c.getJson());
    }
    json.add("corrections", elements);
    assert null != NNLayer.fromJson(json) : "Smoke test deserialization";
    return json;
  }
  
  /**
   * New bias nn layer.
   *
   * @param dims   the dims
   * @param weight the weight
   * @return the nn layer
   */
  public NNLayer newBias(final int[] dims, final double weight) {
    return new BiasLayer(dims).setWeights(i -> weight);
  }
  
  /**
   * New nth power layer nn layer.
   *
   * @param power the power
   * @return the nn layer
   */
  public NNLayer newNthPowerLayer(final double power) {
    return new NthPowerActivationLayer().setPower(power);
  }
  
  /**
   * New product layer nn layer.
   *
   * @return the nn layer
   */
  public NNLayer newProductLayer() {
    return new ProductInputsLayer();
  }
  
  /**
   * New synapse nn layer.
   *
   * @param weight the weight
   * @return the nn layer
   */
  public NNLayer newSynapse(final double weight) {
    return new FullyConnectedLayer(inputDims, outputDims).setWeights(() -> weight * (Math.random() - 1));
  }
  
  /**
   * The type Correcton.
   */
  public class Correcton {
    /**
     * The Bias.
     */
    public final NNLayer bias;
    /**
     * The Factor.
     */
    public final NNLayer factor;
    /**
     * The Power.
     */
    public final double power;

    /**
     * Instantiates a new Correcton.
     *
     * @param power  the power
     * @param bias   the bias
     * @param factor the factor
     */
    public Correcton(final double power, final NNLayer bias, final NNLayer factor) {
      this.power = power;
      this.bias = bias;
      this.factor = factor;
    }

    /**
     * Instantiates a new Correcton.
     *
     * @param json the json
     */
    public Correcton(final JsonObject json) {
      power = json.get("power").getAsDouble();
      bias = layersById.get(UUID.fromString(json.get("bias").getAsString()));
      factor = layersById.get(UUID.fromString(json.get("factor").getAsString()));
    }

    /**
     * Add dag node.
     *
     * @param input the input
     * @return the dag node
     */
    public DAGNode add(final DAGNode input) {
      return PolynomialNetwork.this.add(newNthPowerLayer(power), PolynomialNetwork.this.add(bias, PolynomialNetwork.this.add(factor, input)));
    }
  
    /**
     * Gets json.
     *
     * @return the json
     */
    public JsonObject getJson() {
      final JsonObject json = new JsonObject();
      json.addProperty("bias", bias.getId().toString());
      json.addProperty("factor", factor.getId().toString());
      json.addProperty("power", power);
      return json;
    }
  }
  
}
