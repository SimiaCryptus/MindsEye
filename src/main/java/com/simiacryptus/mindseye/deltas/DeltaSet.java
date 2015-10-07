package com.simiacryptus.mindseye.deltas;

import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.math.VectorLogic;
import com.simiacryptus.mindseye.net.NNLayer;

public class DeltaSet implements VectorLogic<DeltaSet> {
  public final java.util.concurrent.ConcurrentHashMap<NNLayer<?>, DeltaBuffer> map = new java.util.concurrent.ConcurrentHashMap<>();

  public DeltaSet() {
  }

  public DeltaSet(final Map<NNLayer<?>, DeltaBuffer> collect) {
    this.map.putAll(collect);
  }

  @Override
  public DeltaSet add(final DeltaSet right) {
    return join(right, (l, r) -> l.add(r));
  }

  @Override
  public double dotProduct(final DeltaSet right) {
    return sum(right, (l, r) -> l.dotProduct(r));
  }

  public DeltaBuffer get(final NNLayer<?> layer, final double[] ptr) {
    return this.map.computeIfAbsent(layer, l -> new DeltaBuffer(ptr, layer));
  }

  public DeltaBuffer get(final NNLayer<?> layer, final NDArray ptr) {
    return get(layer, ptr.getData());
  }

  public DeltaSet join(final DeltaSet right, final BiFunction<DeltaBuffer, DeltaBuffer, DeltaBuffer> joiner) {
    final HashSet<NNLayer<?>> keys = new HashSet<>(this.map.keySet());
    keys.addAll(right.map.keySet());
    return new DeltaSet(keys.stream().collect(Collectors.toMap(k -> k, k -> {
      final DeltaBuffer l = this.map.get(k);
      final DeltaBuffer r = right.map.get(k);
      if (null != l && null != r)
        return joiner.apply(l, r);
      if (null != l)
        return r;
      if (null != r)
        return l;
      return null;
    })));
  }

  @Override
  public double l1() {
    return Math.sqrt(this.map.values().stream().mapToDouble(v -> v.l1()).sum());
  }

  @Override
  public double l2() {
    return Math.sqrt(this.map.values().stream().mapToDouble(v -> {
      final double l2 = v.l2();
      return l2 * l2;
    }).sum());
  }

  public DeltaSet map(final Function<DeltaBuffer, DeltaBuffer> mapper) {
    return new DeltaSet(this.map.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue()))));
  }

  @Override
  public DeltaSet scale(final double f) {
    return map(x -> x.scale(f));
  }

  public double sum(final DeltaSet right, final BiFunction<DeltaBuffer, DeltaBuffer, Double> joiner) {
    final HashSet<NNLayer<?>> keys = new HashSet<>(this.map.keySet());
    keys.addAll(right.map.keySet());
    return keys.stream().mapToDouble(k -> {
      final DeltaBuffer l = this.map.get(k);
      final DeltaBuffer r = right.map.get(k);
      if (null != l && null != r)
        return joiner.apply(l, r);
      return 0;
    }).sum();
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
