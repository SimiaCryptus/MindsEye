package com.simiacryptus.mindseye.net.dag;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.NNResult;

import java.util.UUID;

/**
 * Created by Andrew Charneski on 4/25/2017.
 */
abstract class LazyResult implements DAGNode {

  public final UUID key;

  public LazyResult() {
    this(UUID.randomUUID());
  }

  protected LazyResult(final UUID key) {
    super();
    this.key = key;
  }

  protected abstract NNResult eval(EvaluationContext t);

  @Override
  public NNResult get(final EvaluationContext t) {
    return t.cache.computeIfAbsent(this.key, k -> eval(t));
  }

  public abstract JsonObject toJson();

}
