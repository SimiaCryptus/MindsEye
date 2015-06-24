package com.simiacryptus.mindseye;

import java.util.function.Function;

public class LabeledObject<T> {
  public final T data;
  public final String label;
  public LabeledObject(T img, String name) {
    super();
    this.data = img;
    this.label = name;
  }
  
  public <U> LabeledObject<U> map(Function<T, U> f) {
    return new LabeledObject<U>(f.apply(data), label);
  }
  
}