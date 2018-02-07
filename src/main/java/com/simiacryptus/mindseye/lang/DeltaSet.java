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

import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * This is a collection of Deltas being staged for particular layers. Provides indexing capabilities to reference the
 * deltas based on physical references (to double[] objects) and based on logical referants (i.e. layers) Provides
 * collection-arithmetic operations appropriate to the Delta's vector geometric archtype.
 *
 * @param <K> the type parameter
 */
public class DeltaSet<K extends ReferenceCounting> extends DoubleBufferSet<K, Delta<K>> {
  
  /**
   * Instantiates a new Delta setByCoord.
   */
  public DeltaSet() {
  }
  
  /**
   * Instantiates a new Delta setByCoord.
   *
   * @param toCopy the to copy
   */
  public DeltaSet(final DoubleBufferSet<K, Delta<K>> toCopy) {
    super(toCopy);
    assert stream().allMatch(x -> x instanceof Delta);
  }
  
  /**
   * Instantiates a new Delta setByCoord.
   *
   * @param collect the collect
   */
  public DeltaSet(final Map<K, ? extends Delta<K>> collect) {
    super(collect);
    assert stream().allMatch(x -> x instanceof Delta);
  }
  
  /**
   * Accumulate delta setByCoord.
   *
   * @param alpha the alpha
   * @return the delta setByCoord
   */
  public @NotNull DeltaSet<K> accumulate(final double alpha) {
    stream().forEach(d -> d.accumulate(alpha));
    return this;
  }
  
  
  /**
   * Add delta setByCoord.
   *
   * @param right the right
   * @return the delta setByCoord
   */
  public @NotNull DeltaSet<K> add(final @NotNull DeltaSet<K> right) {
    return this.copy().addInPlace(right);
  }
  
  /**
   * Add in place delta setByCoord.
   *
   * @param right the right
   * @return the delta setByCoord
   */
  public @NotNull DeltaSet<K> addInPlace(final @NotNull DeltaSet<K> right) {
    right.map.forEach(100, (layer, buffer) -> {
      get(layer, buffer.target).addInPlace(buffer).freeRef();
    });
    return this;
  }
  
  /**
   * As state state setByCoord.
   *
   * @return the state setByCoord
   */
  public @NotNull StateSet<K> asState() {
    final @NotNull StateSet<K> returnValue = new StateSet<>();
    map.forEach((layer, delta) -> {
      delta.assertAlive();
      State<K> kState = returnValue.get(layer, delta.target);
      kState.set(delta.delta);
      kState.freeRef();
    });
    return returnValue;
  }
  
  @Override
  public @NotNull DeltaSet<K> copy() {
    return this.map(x -> x.copy());
  }
  
  /**
   * Dot double.
   *
   * @param right the right
   * @return the double
   */
  public double dot(final @NotNull DoubleBufferSet<K, Delta<K>> right) {
    Stream<Map.Entry<K, Delta<K>>> stream = map.entrySet().stream();
    if (100 < map.size()) {
      stream = stream.parallel();
    }
    return stream.mapToDouble(entry -> {
      final K key = entry.getKey();
      final Delta<K> value = entry.getValue();
      final Delta<K> rValue = right.map.get(key);
      if (null != rValue) {
        return value.dot(rValue);
      }
      else {
        return 0;
      }
    }).sum();
  }
  
  @Override
  protected @NotNull Delta<K> factory(final K layer, final double[] target) {
    return new Delta<K>(layer, target);
  }
  
  /**
   * Gets magnitude.
   *
   * @return the magnitude
   */
  public double getMagnitude() {
    Stream<Map.Entry<K, Delta<K>>> stream = map.entrySet().stream();
    if (100 < map.size()) {
      stream = stream.parallel();
    }
    final double[] elementArray = stream.mapToDouble(entry -> {
      final DoubleBuffer<K> value = entry.getValue();
      final double v = value.deltaStatistics().sumSq();
      return v;
    }).toArray();
    return Math.sqrt(Arrays.stream(elementArray).sum());
  }
  
  @Override
  public @NotNull DeltaSet<K> map(final Function<Delta<K>, Delta<K>> mapper) {
    DoubleBufferSet<K, Delta<K>> map = super.map(mapper);
    @NotNull DeltaSet<K> kDeltaSet = new DeltaSet<>(map);
    map.freeRef();
    return kDeltaSet;
  }
  
  /**
   * Scale delta setByCoord.
   *
   * @param f the f
   * @return the delta setByCoord
   */
  public @NotNull DeltaSet<K> scale(final double f) {
    return map(x -> x.scale(f));
  }
  
  /**
   * Subtract delta setByCoord.
   *
   * @param right the right
   * @return the delta setByCoord
   */
  public @NotNull DeltaSet<K> subtract(final DeltaSet<K> right) {
    return this.add(new DeltaSet<K>(right).scale(-1));
  }
  
  /**
   * Unit delta setByCoord.
   *
   * @return the delta setByCoord
   */
  public @NotNull DeltaSet<K> unit() {
    return scale(1.0 / getMagnitude());
  }
  
}

