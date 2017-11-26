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
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A collection of DoubleBuffer objects being staged for particular layers.
 * Provides indexing capabilities to reference the deltas based on physical references (to double[] objects)
 * and based on logical referants (i.e. layers)
 *
 * @param <T> the type parameter
 */
public abstract class DoubleBufferSet<T extends DoubleBuffer> {
  /**
   * The Map.
   */
  protected final ConcurrentHashMap<NNLayer, T> map = new ConcurrentHashMap<>();
  
  /**
   * Instantiates a new Delta set.
   */
  public DoubleBufferSet() {
  }
  
  /**
   * Instantiates a new Delta set.
   *
   * @param toCopy the to copy
   */
  public DoubleBufferSet(DoubleBufferSet<T> toCopy) {
    this(toCopy.map);
  }
  
  /**
   * Instantiates a new Delta set.
   *
   * @param collect the collect
   */
  public DoubleBufferSet(final Map<NNLayer, ? extends T> collect) {
    map.putAll(collect);
  }
  
  /**
   * Get delta.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the delta
   */
  public T get(final NNLayer layer, final double[] ptr) {
    T delta = get(layer, () -> factory(layer, ptr));
    assert delta.layer.equals(layer);
    assert delta.target == ptr;
    return delta;
  }
  
  /**
   * Factory t.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the t
   */
  protected abstract T factory(final NNLayer layer, final double[] ptr);
  
  /**
   * Get t.
   *
   * @param layer   the layer
   * @param factory the factory
   * @return the t
   */
  public T get(final NNLayer layer, Supplier<T> factory) {
    if (null == map) throw new IllegalArgumentException();
    if (null == factory) throw new IllegalArgumentException();
    if (null == layer) throw new IllegalArgumentException();
    return map.computeIfAbsent(layer, l -> factory.get());
  }
  
  /**
   * Get delta.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the delta
   */
  public T get(final NNLayer layer, final Tensor ptr) {
    return get(layer, ptr.getData());
  }
  
  /**
   * Map delta set.
   *
   * @param mapper the mapper
   * @return the delta set
   */
  public DoubleBufferSet<T> map(final Function<T, T> mapper) {
    DoubleBufferSet<T> parent = this;
    Stream<Map.Entry<NNLayer, T>> stream = map.entrySet().stream();
    if(map.size() > 100) stream = stream.parallel();
    Map<NNLayer, T> newMap = stream.collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue())));
    return new Delegate(parent, newMap);
  }
  
  
  /**
   * Stream stream.
   *
   * @return the stream
   */
  public Stream<T> stream() {
    return map.values().stream().filter(n -> null != n).distinct().sorted(Comparator.comparing(y -> y.getId()));
  }
  
  /**
   * Gets magnitude.
   *
   * @return the magnitude
   */
  public double getMagnitude() {
    double sumSq = map.entrySet().stream().mapToDouble(entry -> {
      DoubleBuffer value = entry.getValue();
      return value.deltaStatistics().sumSq();
    }).sum();
    return Math.sqrt(sumSq);
  }
  
  /**
   * Dot double.
   *
   * @param <U>   the type parameter
   * @param right the right
   * @return the double
   */
  public <U extends Delta> double dot(DoubleBufferSet<U> right) {
    ConcurrentHashMap<NNLayer, U> r = right.map;
    Stream<Map.Entry<NNLayer, T>> stream = map.entrySet().stream();
    if(100 < map.size()) stream = stream.parallel();
    return stream.mapToDouble(entry -> {
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
  
  /**
   * Copy delta set.
   *
   * @return the delta set
   */
  public DoubleBufferSet<T> copy() {
    return map(x -> (T) x.copy());
  }
  
  /**
   * Is different boolean.
   *
   * @return the boolean
   */
  public boolean isDifferent() {
    return stream().parallel().anyMatch(x -> !x.areEqual());
  }
  
  /**
   * Gets map.
   *
   * @return the map
   */
  public ConcurrentHashMap<NNLayer, T> getMap() {
    return map;
  }
  
  /**
   * The type Delegate.
   */
  protected class Delegate extends DoubleBufferSet<T> {
    private final DoubleBufferSet<T> parent;
  
    /**
     * Instantiates a new Delegate.
     *
     * @param parent the parent
     * @param newMap the new map
     */
    public Delegate(DoubleBufferSet<T> parent, Map<NNLayer, T> newMap) {
      super(newMap);
      this.parent = parent;
    }
  
    /**
     * Instantiates a new Delegate.
     *
     * @param parent the parent
     */
    public Delegate(DoubleBufferSet<T> parent) {
      this(parent, new HashMap<>());
    }
  
    @Override
    protected T factory(NNLayer layer, double[] ptr) {
      return parent.factory(layer,ptr);
    }
  }
}
