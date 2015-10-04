package com.simiacryptus.mindseye.net.dag;

import java.util.UUID;
import java.util.function.Function;

import com.google.gson.JsonObject;

public abstract class LazyResult<T> {

  public final UUID key = UUID.randomUUID();

  public LazyResult() {
    super();
  }

  protected abstract T eval(EvaluationContext t);

  @SuppressWarnings("unchecked")
  public T get(final EvaluationContext t) {
    return (T) t.cache.computeIfAbsent(this.key, k -> eval(t));
  }

  protected abstract JsonObject toJson();

  public <U> LazyResult<U> map(Function<T,U> f) {
    LazyResult<T> prev = this;
    return new LazyResult<U>() {
      @Override
      protected U eval(EvaluationContext t) {
        return f.apply(prev.get(t));
      }

      @Override
      protected JsonObject toJson() {
        return null;
      }
    };
  }
}
