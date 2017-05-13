package com.simiacryptus.mindseye.graph.dag;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

import java.util.UUID;

final class InputNode extends LazyResult {
  private DAGNetwork dagNetwork;
  public final UUID handle;

  InputNode(DAGNetwork dagNetwork) {
    this(dagNetwork,null);
  }

  public InputNode(DAGNetwork dagNetwork, final UUID handle) {
    super(handle);
    this.dagNetwork = dagNetwork;
    this.handle = handle;
  }

  @Override
  protected NNResult eval(final EvaluationContext t) {
    return t.cache.get(this.handle);
  }

  @Override
  public JsonObject toJson() {
    final JsonObject json = new JsonObject();
    json.addProperty("target", dagNetwork.inputHandles.toString());
    return json;
  }

  @Override
  public UUID getId() {
    return handle;
  }

  @Override
  public NNLayer getLayer() {
    return null;
  }

  public DAGNode add(NNLayer nextHead) {
    return dagNetwork.add(nextHead, InputNode.this);
  }
}
