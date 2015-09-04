package com.simiacryptus.mindseye.deltas;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.math.NDArray;

public class DeltaBuffer implements VectorLogic<DeltaBuffer> {
  public final Map<NNLayer, DeltaFlushBuffer> map = new LinkedHashMap<>();

  public DeltaBuffer() {
  }

  public DeltaBuffer(Map<NNLayer, DeltaFlushBuffer> collect) {
    map.putAll(collect);
  }

  public synchronized DeltaFlushBuffer get(final NNLayer layer, final double[] ptr) {
    return this.map.computeIfAbsent(layer, l -> new DeltaFlushBuffer(ptr, layer));
  }
  
  public DeltaFlushBuffer get(final NNLayer layer, final NDArray ptr) {
    return get(layer, ptr.getData());
  }

  @Override
  public DeltaBuffer scale(double f) {
    return map(x->x.scale(f));
  }

  @Override
  public double dotProduct(DeltaBuffer right) {
    return sum(right, (l,r)-> l.dotProduct(r));
  }

  @Override
  public DeltaBuffer add(DeltaBuffer right) {
    return join(right, (l,r)-> l.add(r));
  }

  public DeltaBuffer join(DeltaBuffer right, BiFunction<DeltaFlushBuffer, DeltaFlushBuffer, DeltaFlushBuffer> joiner) {
    HashSet<NNLayer> keys = new HashSet<>(map.keySet());
    keys.addAll(right.map.keySet());
    return new DeltaBuffer(keys.stream().collect(Collectors.toMap(k->k, k->{
      DeltaFlushBuffer l = map.get(k);
      DeltaFlushBuffer r = right.map.get(k);
      if(null!=l&&null!=r) return joiner.apply(l, r);
      if(null!=l) return r;
      if(null!=r) return l;
      return null;
    })));
  }

  public double sum(DeltaBuffer right, BiFunction<DeltaFlushBuffer, DeltaFlushBuffer, Double> joiner) {
    HashSet<NNLayer> keys = new HashSet<>(map.keySet());
    keys.addAll(right.map.keySet());
    return keys.stream().mapToDouble(k->{
      DeltaFlushBuffer l = map.get(k);
      DeltaFlushBuffer r = right.map.get(k);
      if(null!=l&&null!=r) return joiner.apply(l, r);
      return 0;
    }).sum();
  }

  public DeltaBuffer map(Function<DeltaFlushBuffer, DeltaFlushBuffer> mapper) {
    return new DeltaBuffer(map.entrySet().stream().collect(Collectors.toMap(e->e.getKey(), e->mapper.apply(e.getValue()))));
  }

  @Override
  public double l2() {
    return Math.sqrt(this.map.values().stream().mapToDouble(v->{
      double l2 = v.l2();
      return l2*l2;
    }).sum());
  }

  @Override
  public double l1() {
    return Math.sqrt(this.map.values().stream().mapToDouble(v->v.l1()).sum());
  }
}
