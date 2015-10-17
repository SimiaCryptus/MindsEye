package com.simiacryptus.mindseye.core.delta;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.simiacryptus.mindseye.core.NDArray;

public class DeltaSet {
  public final java.util.concurrent.ConcurrentHashMap<NNLayer<?>, DeltaBuffer> map = new java.util.concurrent.ConcurrentHashMap<>();

  public DeltaSet() {
  }

  public DeltaSet(final Map<NNLayer<?>, DeltaBuffer> collect) {
    this.map.putAll(collect);
  }

  public DeltaBuffer get(final NNLayer<?> layer, final double[] ptr) {
    return this.map.computeIfAbsent(layer, l -> new DeltaBuffer(ptr, layer));
  }

  public DeltaBuffer get(final NNLayer<?> layer, final NDArray ptr) {
    return get(layer, ptr.getData());
  }

  public DeltaSet map(final Function<DeltaBuffer, DeltaBuffer> mapper) {
    return new DeltaSet(this.map.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue()))));
  }

  public DeltaSet scale(final double f) {
    return map(x -> x.scale(f));
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append("DeltaBuffer [");
    builder.append(vector());
    builder.append("]");
    return builder.toString();
  }

  public List<DeltaBuffer> vector() {
    return this.map.values().stream().filter(n -> null != n).distinct().sorted(Comparator.comparing(y -> y.getId())).collect(Collectors.toList());
  }

}
