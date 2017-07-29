package com.simiacryptus.mindseye.layers;

import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.stream.Stream;

public class TensorArray implements TensorList {
  private final Tensor[] data;
  public TensorArray(Tensor... data) {
    this.data = data;
  }
  @Override
  public Tensor get(int i) { return data[i]; }
  @Override
  public int length() { return data.length; }
  @Override
  public Stream<Tensor> stream() { return Arrays.stream(data); }
}
