package com.simiacryptus.mindseye.deltas;

import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.math.VectorLogic;
import com.simiacryptus.mindseye.net.NNLayer;

public class DeltaBuffer implements VectorLogic<DeltaBuffer> {
  private final Map<NNLayer<?>, DeltaFlushBuffer> map = new LinkedHashMap<>();

  public DeltaBuffer() {
  }

  public DeltaBuffer(final Map<NNLayer<?>, DeltaFlushBuffer> collect) {
    this.map.putAll(collect);
  }

  @Override
  public DeltaBuffer add(final DeltaBuffer right) {
    return join(right, (l, r) -> l.add(r));
  }

  @Override
  public double dotProduct(final DeltaBuffer right) {
    return sum(right, (l, r) -> l.dotProduct(r));
  }

  public synchronized DeltaFlushBuffer get(final NNLayer<?> layer, final double[] ptr) {
    return this.map.computeIfAbsent(layer, l -> new DeltaFlushBuffer(ptr, layer));
  }

  public DeltaFlushBuffer get(final NNLayer<?> layer, final NDArray ptr) {
    return get(layer, ptr.getData());
  }

  public DeltaBuffer join(final DeltaBuffer right, final BiFunction<DeltaFlushBuffer, DeltaFlushBuffer, DeltaFlushBuffer> joiner) {
    final HashSet<NNLayer<?>> keys = new HashSet<>(this.map.keySet());
    keys.addAll(right.map.keySet());
    return new DeltaBuffer(keys.stream().collect(Collectors.toMap(k -> k, k -> {
      final DeltaFlushBuffer l = this.map.get(k);
      final DeltaFlushBuffer r = right.map.get(k);
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

  public DeltaBuffer map(final Function<DeltaFlushBuffer, DeltaFlushBuffer> mapper) {
    return new DeltaBuffer(this.map.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue()))));
  }

  @Override
  public DeltaBuffer scale(final double f) {
    return map(x -> x.scale(f));
  }

  public double sum(final DeltaBuffer right, final BiFunction<DeltaFlushBuffer, DeltaFlushBuffer, Double> joiner) {
    final HashSet<NNLayer<?>> keys = new HashSet<>(this.map.keySet());
    keys.addAll(right.map.keySet());
    return keys.stream().mapToDouble(k -> {
      final DeltaFlushBuffer l = this.map.get(k);
      final DeltaFlushBuffer r = right.map.get(k);
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

  public List<DeltaFlushBuffer> vector() {
    return this.map.values().stream().filter(n -> null != n).distinct().sorted(Comparator.comparing(y -> y.getId())).collect(Collectors.toList());
  }

}
