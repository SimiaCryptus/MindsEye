/*
 * Copyright (c) 2018 by Andrew Charneski.
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

import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A collection of DoubleBuffer objects being staged for particular layers. Provides indexing capabilities to reference
 * the deltas based on physical references (to double[] objects) and based on logical referants (i.e. layers)
 *
 * @param <K> the type parameter
 * @param <T> the type parameter
 */
public abstract class DoubleBufferSet<K extends ReferenceCounting, T extends DoubleBuffer<K>> extends ReferenceCountingBase {
  /**
   * The Log.
   */
  static final Logger log = LoggerFactory.getLogger(SimpleConvolutionLayer.class);
  
  /**
   * The Map.
   */
  @Nonnull
  protected final ConcurrentHashMap<K, T> map = new ConcurrentHashMap<>();
  
  /**
   * Instantiates a new Delta setByCoord.
   */
  public DoubleBufferSet() {
  }
  
  /**
   * Instantiates a new Delta setByCoord.
   *
   * @param toCopy the to copy
   */
  public DoubleBufferSet(@Nonnull final DoubleBufferSet<K, T> toCopy) {
    this(toCopy.map);
  }
  
  /**
   * Instantiates a new Delta setByCoord.
   *
   * @param collect the collect
   */
  public DoubleBufferSet(@Nonnull final Map<K, ? extends T> collect) {
    map.putAll(collect);
    map.forEach((k, v) -> {
      k.addRef(this);
      v.addRef(this);
    });
  }
  
  /**
   * Copy delta setByCoord.
   *
   * @return the delta setByCoord
   */
  @Nonnull
  @SuppressWarnings("unchecked")
  public DoubleBufferSet<K, T> copy() {
    return map(x -> (T) x.copy());
  }
  
  /**
   * Factory t.
   *
   * @param layer  the layer
   * @param target the target
   * @return the t
   */
  protected abstract T factory(final K layer, final double[] target);
  
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
  private T get(@Nullable final K layer, @Nullable final Supplier<T> factory) {
    if (null == map) throw new IllegalArgumentException();
    if (null == factory) throw new IllegalArgumentException();
    if (null == layer) throw new IllegalArgumentException();
    synchronized (map) {
      T v = map.computeIfAbsent(layer, l -> {
        l.addRef(this);
        T delta = factory.get();
        if (log.isDebugEnabled())
          log.debug(String.format("Init layer buffer for %s - %s params", l.getClass(), delta.target.length));
        return delta;
      });
      v.addRef();
      return v;
    }
  }
  
  /**
   * Get delta.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the delta
   */
  public T get(final K layer, @Nonnull final Tensor ptr) {
    return get(layer, ptr.getData());
  }
  
  /**
   * Gets map.
   *
   * @return the map
   */
  @Nonnull
  public ConcurrentHashMap<K, T> getMap() {
    return map;
  }
  
  /**
   * Map delta setByCoord.
   *
   * @param mapper the mapper
   * @return the delta setByCoord
   */
  @Nonnull
  public DoubleBufferSet<K, T> map(@Nonnull final Function<T, T> mapper) {
    @Nonnull final DoubleBufferSet<K, T> parent = this;
    Stream<Map.Entry<K, T>> stream = map.entrySet().stream();
    if (map.size() > 100) {
      stream = stream.parallel();
    }
    final Map<K, T> newMap = stream.collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue())));
    @Nonnull Delegate delegate = new Delegate(parent, newMap);
    newMap.values().forEach(x -> x.freeRef());
    return delegate;
  }
  
  /**
   * Stream stream.
   *
   * @return the stream
   */
  public Stream<T> stream() {
    return map.values().stream().filter(n -> null != n).distinct().sorted(Comparator.comparing(y -> System.identityHashCode(y.target)));
  }
  
  @Override
  protected void _free() {
    map.forEach((k, v) -> {
      k.freeRef();
      v.freeRef();
    });
    map.clear();
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
    public Delegate(final DoubleBufferSet<K, T> parent, @Nonnull final Map<K, T> newMap) {
      super(newMap);
      this.parent = parent;
    }
    
    @Override
    protected T factory(final K layer, final double[] target) {
      return parent.factory(layer, target);
    }
  }
}
