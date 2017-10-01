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
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.activation.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.layers.reducers.ProductInputsLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
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
    Tuning,
    Final
  }
  NNLayer alpha = null;
  NNLayer gamma = null;
  NNLayer beta = null;
  private DAGNode head = null;
  private NodeMode mode = null;
  
  public JsonObject getJson() {
    assertConsistent();
    DAGNode head = getHead();
    JsonObject json = super.getJson();
    json.addProperty("head", head.getId().toString());
    if(null != alpha) json.addProperty("alpha", alpha.getId().toString());
    if(null != beta) json.addProperty("beta", beta.getId().toString());
    if(null != gamma) json.addProperty("gamma", gamma.getId().toString());
    json.addProperty("mode", mode.name());
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
    if(json.get("beta") != null) beta = layersById.get(UUID.fromString(json.get("beta").getAsString()));
    if(json.get("gamma") != null) gamma = layersById.get(UUID.fromString(json.get("gamma").getAsString()));
    mode = NodeMode.valueOf(json.get("mode").getAsString());
  }
  
  /**
   * Instantiates a new Pipeline network.
   */
  public SigmoidTreeNetwork(NNLayer alpha) {
    super(1);
    this.alpha = alpha;
    this.mode = NodeMode.Linear;
  }
  
  public synchronized DAGNode getHead() {
    if(null == head) {
      synchronized (this) {
        if(null == head) {
          this.reset();
          DAGNode input = getInput(0);
          switch (mode) {
            case Linear:
              head = add(alpha.setFrozen(false), input);
              break;
            case Fuzzy:
              head = add(new ProductInputsLayer(),
                add(alpha.setFrozen(false), input),
                add(new SigmoidActivationLayer().setBalanced(false), add(gamma.setFrozen(false), input))
              );
              break;
            case Bilinear: {
              DAGNode gammaNode = add(gamma.setFrozen(true), input);
              head = add(new SumInputsLayer(),
                add(new ProductInputsLayer(),
                  add(alpha.setFrozen(false), input),
                  add(new SigmoidActivationLayer().setBalanced(false), gammaNode)
                ),
                add(new ProductInputsLayer(),
                  add(beta.setFrozen(false), input),
                  add(new SigmoidActivationLayer().setBalanced(false),
                    add(new LinearActivationLayer().setScale(-1).freeze(), gammaNode))
                ));
              break;
            }
            case Tuning: {
              DAGNode gammaNode = add(gamma.setFrozen(false), input);
              head = add(new SumInputsLayer(),
                add(new ProductInputsLayer(),
                  add(alpha.setFrozen(false), input),
                  add(new SigmoidActivationLayer().setBalanced(false), gammaNode)
                ),
                add(new ProductInputsLayer(),
                  add(beta.setFrozen(false), input),
                  add(new SigmoidActivationLayer().setBalanced(false),
                    add(new LinearActivationLayer().setScale(-1).freeze(), gammaNode))
                ));
              break;
            }
            case Final: {
              DAGNode gammaNode = add(gamma.setFrozen(false), input);
              head = add(new SumInputsLayer(),
                add(new ProductInputsLayer(),
                  add(alpha, input),
                  add(new SigmoidActivationLayer().setBalanced(false), gammaNode)
                ),
                add(new ProductInputsLayer(),
                  add(beta, input),
                  add(new SigmoidActivationLayer().setBalanced(false),
                    add(new LinearActivationLayer().setScale(-1).freeze(), gammaNode))
                ));
              break;
            }
          }
        }
      }
    }
    return head;
  }
  
  @Override
  public void nextPhase() {
    switch (mode) {
      case Linear: {
        head = null;
        DenseSynapseLayer alpha = (DenseSynapseLayer) this.alpha;
        gamma = new DenseSynapseLayer(alpha.inputDims, new int[]{1});
        ((DenseSynapseLayer)gamma).setWeights(()->0.01*Math.random());
        mode = NodeMode.Fuzzy;
        break;
      }
      case Fuzzy: {
        head = null;
        DenseSynapseLayer alpha = (DenseSynapseLayer) this.alpha;
        beta = new DenseSynapseLayer(alpha.inputDims, alpha.outputDims);
        List<double[]> alphaState = alpha.state();
        List<double[]> betaState = beta.state();
        for(int i=0;i<alphaState.size();i++) {
          double[] betaBuffer = betaState.get(i);
          double[] alphaBuffer = alphaState.get(i);
          System.arraycopy(alphaBuffer, 0, betaBuffer, 0, alphaBuffer.length);
        }
        mode = NodeMode.Bilinear;
        break;
      }
      case Bilinear: {
        head = null;
        mode = NodeMode.Tuning;
        break;
      }
      case Tuning: {
        head = null;
        alpha = new SigmoidTreeNetwork(alpha);
        ((SigmoidTreeNetwork)alpha).nextPhase();
        beta = new SigmoidTreeNetwork(beta);
        ((SigmoidTreeNetwork)beta).nextPhase();
        mode = NodeMode.Final;
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
  
}
