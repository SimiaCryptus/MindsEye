package com.simiacryptus.mindseye.net.dag;

import com.google.gson.JsonObject;
import com.simiacryptus.util.Util;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

import java.util.UUID;

public final class InnerNode extends LazyResult {
  public final NNLayer<?> nnlayer;
  private DAGNetwork dagNetwork;
  @SuppressWarnings("unused")
  public final String[] createdBy = Util.currentStack();
  public final UUID layer;
  private final DAGNode[] inputNodes;

  @Override
  public DAGNode[] getInputs() {
    return inputNodes;
  };

  @SafeVarargs
  InnerNode(DAGNetwork dagNetwork, final NNLayer<?> layer, final DAGNode... inputNodes) {
    this.dagNetwork = dagNetwork;
    assert null != inputNodes;
    this.layer = layer.getId();
    this.nnlayer = layer;
    this.inputNodes = inputNodes;
  }

  @Override
  protected NNResult eval(final EvaluationContext ctx) {
    if (1 == this.inputNodes.length) {
      final NNResult in = this.inputNodes[0].get(ctx);
      final NNResult output = dagNetwork.byId.get(this.layer).eval(in);
      return output;
    } else {
      final NNResult[] in = java.util.Arrays.stream(this.inputNodes).map(x -> ((LazyResult) x).get(ctx)).toArray(i -> new NNResult[i]);
      final NNResult output = dagNetwork.byId.get(this.layer).eval(in);
      return output;
    }
  }

  @Override
  public JsonObject toJson() {
    final JsonObject json = new JsonObject();
    json.add("layer", dagNetwork.byId.get(this.layer).getJson());
    if (this.inputNodes.length > 0) json.add("prev0", ((LazyResult) this.inputNodes[0]).toJson());
    return json;
  }

  @Override
  public UUID getId() {
    return this.layer;
  }

  @Override
  public DAGNode add(NNLayer<?> nextHead) {
    return dagNetwork.add(nextHead, InnerNode.this).getHead();
  }
}
