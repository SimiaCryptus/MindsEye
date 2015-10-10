package com.simiacryptus.mindseye.core;

import java.util.function.Function;

public class LabeledObject<T> {
  public final T data;
  public final String label;

  public LabeledObject(final T img, final String name) {
    super();
    this.data = img;
    this.label = name;
  }

  public <U> LabeledObject<U> map(final Function<T, U> f) {
    return new LabeledObject<U>(f.apply(this.data), this.label);
  }

}
