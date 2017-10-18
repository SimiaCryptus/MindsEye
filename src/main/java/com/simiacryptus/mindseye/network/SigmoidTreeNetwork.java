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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.reducers.ProductInputsLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;

import java.util.List;
import java.util.UUID;

/**
 * The type Pipeline network.
 */
public class SigmoidTreeNetwork extends DAGNetwork implements EvolvingNetwork {
  
  
  public enum NodeMode {
    Linear,
    Fuzzy,
    Bilinear,
    Final
  }
  private NNLayer alpha = null;
  private NNLayer alphaBias = null;
  private NNLayer gate = null;
  private NNLayer gateBias = null;
  private NNLayer beta = null;
  private NNLayer betaBias = null;
  private DAGNode head = null;
  private NodeMode mode = null;
  private boolean skipChildStage = true;
  private boolean multigate = false;
  private boolean skipFuzzy = false;
  
  public JsonObject getJson() {
    assertConsistent();
    DAGNode head = getHead();
    JsonObject json = super.getJson();
    json.addProperty("head", head.getId().toString());
    if(null != alpha) json.addProperty("alpha", alpha.getId().toString());
    if(null != alphaBias) json.addProperty("alphaBias", alpha.getId().toString());
    if(null != beta) json.addProperty("beta", beta.getId().toString());
    if(null != betaBias) json.addProperty("betaBias", beta.getId().toString());
    if(null != gate) json.addProperty("gate", gate.getId().toString());
    if(null != gateBias) json.addProperty("gateBias", gate.getId().toString());
    json.addProperty("mode", getMode().name());
    json.addProperty("skipChildStage", skipChildStage());
    json.addProperty("skipFuzzy", isSkipFuzzy());
    assert null != NNLayer.fromJson(json) : "Smoke test deserialization";
    return json;
  }
  
  /**
   * From json pipeline network.
   *
   * @param json the json
   * @return the pipeline network
   */
  public static SigmoidTreeNetwork fromJson(JsonObject json) {
    return new SigmoidTreeNetwork(json);
  }
  
  /**
   * Instantiates a new Pipeline network.
   *
   * @param json the json
   */
  protected SigmoidTreeNetwork(JsonObject json) {
    super(json);
    head = nodesById.get(UUID.fromString(json.get("head").getAsString()));
    if(json.get("alpha") != null) alpha = layersById.get(UUID.fromString(json.get("alpha").getAsString()));
    if(json.get("alphaBias") != null) alphaBias = layersById.get(UUID.fromString(json.get("alphaBias").getAsString()));
    if(json.get("beta") != null) beta = layersById.get(UUID.fromString(json.get("beta").getAsString()));
    if(json.get("betaBias") != null) betaBias = layersById.get(UUID.fromString(json.get("betaBias").getAsString()));
    if(json.get("gate") != null) gate = layersById.get(UUID.fromString(json.get("gate").getAsString()));
    if(json.get("gateBias") != null) gate = layersById.get(UUID.fromString(json.get("gateBias").getAsString()));
    setSkipChildStage((json.get("skipChildStage") != null) ? json.get("skipChildStage").getAsBoolean() : skipChildStage());
    setSkipFuzzy(((json.get("skipFuzzy") != null) ? json.get("skipFuzzy").getAsBoolean() : isSkipFuzzy()));
    mode = NodeMode.valueOf(json.get("mode").getAsString());
  }
  
  /**
   * Instantiates a new Pipeline network.
   */
  public SigmoidTreeNetwork(NNLayer alpha, NNLayer alphaBias) {
    super(1);
    this.alpha = alpha;
    this.alphaBias = alphaBias;
    this.mode = NodeMode.Linear;
  }
  
  public synchronized DAGNode getHead() {
    if(null == head) {
      synchronized (this) {
        if(null == head) {
          this.reset();
          DAGNode input = getInput(0);
          switch (getMode()) {
            case Linear:
              head = add(alpha.setFrozen(false), add(alphaBias.setFrozen(false), input));
              break;
            case Fuzzy: {
              DAGNode gateNode = add(gate.setFrozen(false), (null != gateBias) ? add(gateBias.setFrozen(false), input) : input);
              head = add(new ProductInputsLayer(),
                add(alpha.setFrozen(false), add(alphaBias.setFrozen(false), input)),
                add(new LinearActivationLayer().setScale(2).freeze(),
                  add(new SigmoidActivationLayer().setBalanced(false), gateNode))
              );
              break;
            }
            case Bilinear: {
              DAGNode gateNode = add(gate.setFrozen(false), (null != gateBias) ? add(gateBias.setFrozen(false), input) : input);
              head = add(new SumInputsLayer(),
                add(new ProductInputsLayer(),
                  add(alpha.setFrozen(false), add(alphaBias.setFrozen(false), input)),
                  add(new SigmoidActivationLayer().setBalanced(false), gateNode)
                ),
                add(new ProductInputsLayer(),
                  add(beta.setFrozen(false), add(betaBias.setFrozen(false), input)),
                  add(new SigmoidActivationLayer().setBalanced(false),
                    add(new LinearActivationLayer().setScale(-1).freeze(), gateNode))
                ));
              break;
            }
            case Final: {
              DAGNode gateNode = add(gate.setFrozen(false), (null != gateBias) ? add(gateBias.setFrozen(false), input) : input);
              head = add(new SumInputsLayer(),
                add(new ProductInputsLayer(),
                  add(alpha, input),
                  add(new SigmoidActivationLayer().setBalanced(false), gateNode)
                ),
                add(new ProductInputsLayer(),
                  add(beta, input),
                  add(new SigmoidActivationLayer().setBalanced(false),
                    add(new LinearActivationLayer().setScale(-1).freeze(), gateNode))
                ));
              break;
            }
          }
        }
      }
    }
    return head;
  }
  
  double initialFuzzyCoeff = 1e-8;
  @Override
  public void nextPhase() {
    switch (getMode()) {
      case Linear: {
        this.head = null;
        DenseSynapseLayer alpha = (DenseSynapseLayer) this.alpha;
        //alpha.weights.scale(2);
        this.gate = new DenseSynapseLayer(alpha.inputDims, multigate?alpha.outputDims:new int[]{1});
        this.gateBias = new BiasLayer(alpha.inputDims);
        this.mode = NodeMode.Fuzzy;
        break;
      }
      case Fuzzy: {
        this.head = null;
        DenseSynapseLayer alpha = (DenseSynapseLayer) this.alpha;
        BiasLayer alphaBias = (BiasLayer) this.alphaBias;
        this.beta = new DenseSynapseLayer(alpha.inputDims, alpha.outputDims).setWeights(() -> {
          return initialFuzzyCoeff * (Math.random() - 0.5);
        });
        this.betaBias = new BiasLayer(alphaBias.bias.length);
        copyState(alpha, beta);
        copyState(alphaBias, betaBias);
        this.mode = NodeMode.Bilinear;
        if(isSkipFuzzy()) nextPhase();
        break;
      }
      case Bilinear: {
        this.head = null;
        this.alpha = new SigmoidTreeNetwork(alpha, alphaBias);
        if(skipChildStage()) ((SigmoidTreeNetwork)alpha).nextPhase();
        this.beta = new SigmoidTreeNetwork(beta, betaBias);
        if(skipChildStage()) ((SigmoidTreeNetwork)beta).nextPhase();
        this.mode = NodeMode.Final;
        break;
      }
      case Final: {
        SigmoidTreeNetwork alpha = (SigmoidTreeNetwork) this.alpha;
        SigmoidTreeNetwork beta = (SigmoidTreeNetwork) this.beta;
        alpha.nextPhase();
        beta.nextPhase();
        break;
      }
    }
  }
  
  public void copyState(NNLayer from, NNLayer to) {
    List<double[]> alphaState = from.state();
    List<double[]> betaState = to.state();
    for(int i=0;i<alphaState.size();i++) {
      double[] betaBuffer = betaState.get(i);
      double[] alphaBuffer = alphaState.get(i);
      System.arraycopy(alphaBuffer, 0, betaBuffer, 0, alphaBuffer.length);
    }
  }
  
  public boolean skipChildStage() {
    return skipChildStage;
  }
  
  public SigmoidTreeNetwork setSkipChildStage(boolean skipChildStage) {
    this.skipChildStage = skipChildStage;
    return this;
  }
  
  public NodeMode getMode() {
    return mode;
  }
  
  public boolean isSkipFuzzy() {
    return skipFuzzy;
  }
  
  public SigmoidTreeNetwork setSkipFuzzy(boolean skipFuzzy) {
    this.skipFuzzy = skipFuzzy;
    return this;
  }
  
}
