package com.simiacryptus.mindseye.net.dag;

import java.util.UUID;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.NNResult;

public abstract class LazyResult {

  public final UUID key;

  public LazyResult() {
    this(UUID.randomUUID());
  }

  protected LazyResult(UUID key) {
    super();
    this.key = key;
  }

  protected abstract NNResult eval(EvaluationContext t);

  public NNResult get(final EvaluationContext t) {
    return (NNResult) t.cache.computeIfAbsent(this.key, k -> eval(t));
  }

  protected abstract JsonObject toJson();

}
