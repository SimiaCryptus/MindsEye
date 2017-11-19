/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.lang;

import java.util.Comparator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Delta set.
 */
public class DeltaSet {
  private final ConcurrentHashMap<NNLayer, Delta> map = new ConcurrentHashMap<>();
  
  /**
   * Instantiates a new Delta set.
   */
  public DeltaSet() {
  }
  
  /**
   * Instantiates a new Delta set.
   *
   * @param collect the collect
   */
  public DeltaSet(final Map<NNLayer, Delta> collect) {
    map.putAll(collect);
  }
  
  /**
   * Get delta buffer.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the delta buffer
   */
  public Delta get(final NNLayer layer, final double[] ptr) {
    Delta delta = get(layer, () -> new Delta(ptr, layer));
    assert delta.layer.equals(layer);
    assert delta.target == ptr;
    return delta;
  }
  
  /**
   * Get t.
   *
   * @param <T>     the type parameter
   * @param layer   the layer
   * @param factory the factory
   * @return the t
   */
  public <T extends Delta> T get(final NNLayer layer, Supplier<T> factory) {
    if (null == map) throw new IllegalArgumentException();
    if (null == factory) throw new IllegalArgumentException();
    if (null == layer) throw new IllegalArgumentException();
    return (T) map.computeIfAbsent(layer, l -> factory.get());
  }
  
  /**
   * Get delta buffer.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the delta buffer
   */
  public Delta get(final NNLayer layer, final Tensor ptr) {
    return get(layer, ptr.getData());
  }
  
  /**
   * Map delta set.
   *
   * @param mapper the mapper
   * @return the delta set
   */
  public DeltaSet map(final Function<Delta, Delta> mapper) {
    return new DeltaSet(map.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue()))));
  }
  
  /**
   * Scale delta set.
   *
   * @param f the f
   * @return the delta set
   */
  public DeltaSet scale(final double f) {
    return map(x -> x.scale(f));
  }
  
  public Stream<Delta> stream() {
    return map.values().stream().filter(n -> null != n).distinct().sorted(Comparator.comparing(y -> y.getId()));
  }
  
  /**
   * Unit delta set.
   *
   * @return the delta set
   */
  public DeltaSet unit() {
    return scale(1.0 / getMagnitude());
  }
  
  /**
   * Gets magnitude.
   *
   * @return the magnitude
   */
  public double getMagnitude() {
    double sumSq = map.entrySet().stream().mapToDouble(entry -> {
      Delta value = entry.getValue();
      return value.sumSq();
    }).sum();
    return Math.sqrt(sumSq);
  }
  
  /**
   * Dot double.
   *
   * @param right the right
   * @return the double
   */
  public double dot(DeltaSet right) {
    ConcurrentHashMap<NNLayer, Delta> r = right.map;
    return map.entrySet().stream().mapToDouble(entry -> {
      NNLayer key = entry.getKey();
      assert key.equals(key);
      if (r.containsKey(key)) {
        return entry.getValue().dot(r.get(key));
      }
      else {
        return 0;
      }
    }).sum();
  }
  
  public DeltaSet subtract(DeltaSet right) {
    return this.add(right.scale(-1));
  }
  
  /**
   * Add delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DeltaSet add(DeltaSet right) {
    DeltaSet returnValue = new DeltaSet();
    map.forEach((layer, buffer) -> {
      returnValue.get(layer, buffer.target)
        .accumulate(buffer.getDelta());
    });
    right.map.forEach((layer, buffer) -> {
      returnValue.get(layer, buffer.target)
        .accumulate(buffer.getDelta());
    });
    return returnValue;
  }
  
  public DeltaSet union(DeltaSet right) {
    return new DeltaSet(Stream.concat(
      map.entrySet().stream(),
      right.map.entrySet().stream()
    ).collect(Collectors.groupingBy(e1 -> e1.getKey(),
      Collectors.mapping(x -> x.getValue(), Collectors.collectingAndThen(
        Collectors.reducing((a, b) -> a.accumulate(b.getDelta())), x -> x.get())))));
  }
  
  /**
   * Copy delta set.
   *
   * @return the delta set
   */
  public DeltaSet copy() {
    return map(x -> x.copy());
  }
  
  /**
   * Write delta set.
   *
   * @param alpha the alpha
   * @return the delta set
   */
  public DeltaSet accumulate(double alpha) {
    stream().forEach(d -> d.accumulate(alpha));
    return this;
  }
  
  /**
   * Write delta set.
   *
   * @return the delta set
   */
  public DeltaSet accumulate() {
    accumulate(1);
    return this;
  }
  
  /**
   * Is different boolean.
   *
   * @return the boolean
   */
  public boolean isDifferent() {
    return stream().parallel().anyMatch(x -> !x.areEqual());
  }
  
  public ConcurrentHashMap<NNLayer, Delta> getMap() {
    return map;
  }
}
