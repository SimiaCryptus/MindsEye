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
 * @param <K> the type parameter
 * @param <T> the type parameter
 */
public abstract class DoubleBufferSet<K, T extends DoubleBuffer<K>> {
  /**
   * The Map.
   */
  protected final ConcurrentHashMap<K, T> map = new ConcurrentHashMap<>();
  
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
  public DoubleBufferSet(final DoubleBufferSet<K, T> toCopy) {
    this(toCopy.map);
  }
  
  /**
   * Instantiates a new Delta set.
   *
   * @param collect the collect
   */
  public DoubleBufferSet(final Map<K, ? extends T> collect) {
    map.putAll(collect);
  }
  
  /**
   * Copy delta set.
   *
   * @return the delta set
   */
  @SuppressWarnings("unchecked")
  public DoubleBufferSet<K, T> copy() {
    return map(x -> (T) x.copy());
  }
  
  /**
   * Factory t.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the t
   */
  protected abstract T factory(final K layer, final double[] ptr);
  
  /**
   * Get delta.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the delta
   */
  public T get(final K layer, final double[] ptr) {
    final T delta = get(layer, () -> factory(layer, ptr));
    assert delta.layer.equals(layer);
    assert delta.target == ptr;
    return delta;
  }
  
  /**
   * Get t.
   *
   * @param layer   the layer
   * @param factory the factory
   * @return the t
   */
  public T get(final K layer, final Supplier<T> factory) {
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
  public T get(final K layer, final Tensor ptr) {
    return get(layer, ptr.getData());
  }
  
  /**
   * Gets map.
   *
   * @return the map
   */
  public ConcurrentHashMap<K, T> getMap() {
    return map;
  }
  
  /**
   * Map delta set.
   *
   * @param mapper the mapper
   * @return the delta set
   */
  public DoubleBufferSet<K, T> map(final Function<T, T> mapper) {
    final DoubleBufferSet<K, T> parent = this;
    Stream<Map.Entry<K, T>> stream = map.entrySet().stream();
    if (map.size() > 100) {
      stream = stream.parallel();
    }
    final Map<K, T> newMap = stream.collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue())));
    return new Delegate(parent, newMap);
  }
  
  /**
   * Stream stream.
   *
   * @return the stream
   */
  public Stream<T> stream() {
    return map.values().stream().filter(n -> null != n).distinct().sorted(Comparator.comparing(y -> System.identityHashCode(y.target)));
  }
  
  /**
   * The type Delegate.
   */
  protected class Delegate extends DoubleBufferSet<K, T> {
    private final DoubleBufferSet<K, T> parent;
  
    /**
     * Instantiates a new Delegate.
     *
     * @param parent the parent
     */
    public Delegate(final DoubleBufferSet<K, T> parent) {
      this(parent, new HashMap<>());
    }
  
    /**
     * Instantiates a new Delegate.
     *
     * @param parent the parent
     * @param newMap the new map
     */
    public Delegate(final DoubleBufferSet<K, T> parent, final Map<K, T> newMap) {
      super(newMap);
      this.parent = parent;
    }
    
    @Override
    protected T factory(final K layer, final double[] ptr) {
      return parent.factory(layer, ptr);
    }
  }
}
