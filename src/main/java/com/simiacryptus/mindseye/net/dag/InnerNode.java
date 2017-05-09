package com.simiacryptus.mindseye.net.dag;

import com.google.gson.JsonObject;
import com.simiacryptus.util.Util;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

import java.util.Arrays;
import java.util.UUID;

public final class InnerNode extends LazyResult {
  public final NNLayer layer;
  private DAGNetwork dagNetwork;
  @SuppressWarnings("unused")
  public final String[] createdBy = Util.currentStack();
  public final UUID id;
  private final DAGNode[] inputNodes;

  @Override
  public DAGNode[] getInputs() {
    return inputNodes;
  }

    @SafeVarargs
  InnerNode(DAGNetwork dagNetwork, final NNLayer id, final DAGNode... inputNodes) {
    this.dagNetwork = dagNetwork;
    assert null != inputNodes;
    this.id = id.getId();
    this.layer = id;
    assert Arrays.stream(inputNodes).allMatch(x->x != null);
    this.inputNodes = inputNodes;
  }

  @Override
  protected NNResult eval(final EvaluationContext ctx) {
    if (1 == this.inputNodes.length) {
      final NNResult in = this.inputNodes[0].get(ctx);
      final NNResult output = dagNetwork.byId.get(this.id).eval(in);
      return output;
    } else {
      final NNResult[] in = java.util.Arrays.stream(this.inputNodes).map(x -> x.get(ctx)).toArray(i -> new NNResult[i]);
      final NNResult output = dagNetwork.byId.get(this.id).eval(in);
      return output;
    }
  }

  @Override
  public JsonObject toJson() {
    final JsonObject json = new JsonObject();
    json.add("id", dagNetwork.byId.get(this.id).getJson());
    if (this.inputNodes.length > 0) json.add("prev0", this.inputNodes[0].toJson());
    return json;
  }

  @Override
  public UUID getId() {
    return this.id;
  }

  @Override
  public NNLayer getLayer() {
    return layer;
  }

  public DAGNode add(NNLayer nextHead) {
    return dagNetwork.add(nextHead, InnerNode.this);
  }
}
