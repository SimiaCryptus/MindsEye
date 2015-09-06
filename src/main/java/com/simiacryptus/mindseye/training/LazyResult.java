package com.simiacryptus.mindseye.training;

import java.util.UUID;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.NNResult;

public abstract class LazyResult {

  UUID key = UUID.randomUUID();

  public LazyResult() {
    super();
  }

  public NNResult[] get(final EvaluationContext t) {
    return t.cache.computeIfAbsent(this.key, k -> eval(t));
  }

  protected abstract NNResult[] eval(EvaluationContext t);

  protected abstract JsonObject toJson();
}