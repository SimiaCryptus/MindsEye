package com.simiacryptus.mindseye.training;

import java.util.HashMap;
import java.util.Map;

public class EvaluationContext {
  
  public static abstract class LazyResult<T> {

    public LazyResult() {
      super();
    }

    @SuppressWarnings("unchecked")
    public T get(EvaluationContext t) {
      return (T) t.cache.computeIfAbsent(this, k->initialValue(t));
    }

    protected abstract T initialValue(EvaluationContext t);
  }
  
  public final Map<Object, Object> cache = new HashMap<>();

}