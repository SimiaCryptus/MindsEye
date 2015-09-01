package com.simiacryptus.mindseye.training;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class EvaluationContext {
  
  public static abstract class LazyResult<T> {

    UUID key = UUID.randomUUID();
    public LazyResult() {
      super();
    }

    @SuppressWarnings("unchecked")
    public T get(EvaluationContext t) {
      return (T) t.cache.computeIfAbsent(key, k->initialValue(t));
    }

    protected abstract T initialValue(EvaluationContext t);
  }
  
  public final Map<UUID, Object> cache = new HashMap<>();

}