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

import java.util.Map;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * This is a collection of Deltas being staged for particular layers.
 * Provides indexing capabilities to reference the deltas based on physical references (to double[] objects)
 * and based on logical referants (i.e. layers)
 * Provides collection-arithmetic operations appropriate to the Delta's vector geometric archtype.
 */
public class DeltaSet<K> extends DoubleBufferSet<K,Delta<K>> {
  
  /**
   * Instantiates a new Delta set.
   */
  public DeltaSet() {
  }
  
  /**
   * Instantiates a new Delta set.
   *
   * @param toCopy the to copy
   */
  public DeltaSet(DoubleBufferSet<K,Delta<K>> toCopy) {
    super(toCopy);
    assert stream().allMatch(x->x instanceof Delta);
  }
  
  /**
   * Instantiates a new Delta set.
   *
   * @param collect the collect
   */
  public DeltaSet(Map<K, ? extends Delta<K>> collect) {
    super(collect);
    assert stream().allMatch(x->x instanceof Delta);
  }
  
  @Override
  protected Delta factory(K layer, double[] ptr) {
    return new Delta(layer, ptr);
  }
  
  
  /**
   * Subtract delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DeltaSet subtract(DeltaSet right) {
    return this.add(new DeltaSet(right).scale(-1));
  }
  
  /**
   * Add delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DeltaSet<K> add(DeltaSet<K> right) {
    return this.copy().addInPlace(right);
  }
  
  private DeltaSet<K> addInPlace(DeltaSet<K> right) {
    right.map.forEach(100, (layer, buffer) -> {
      get(layer, buffer.target).addInPlace(buffer);
    });
    return this;
  }
  
  /**
   * Scale delta set.
   *
   * @param f the f
   * @return the delta set
   */
  public DeltaSet<K> scale(final double f) {
    return new DeltaSet(map(x -> x.scale(f)));
  }
  
  @Override
  public DeltaSet<K> map(Function<Delta<K>, Delta<K>> mapper) {
    return new DeltaSet(super.map(mapper));
  }
  
  @Override
  public DeltaSet<K> copy() {
    return this.map(x->x.copy());
  }
  
  /**
   * Accumulate delta set.
   *
   * @param alpha the alpha
   * @return the delta set
   */
  public DeltaSet<K> accumulate(double alpha) {
    stream().forEach(d -> d.accumulate(alpha));
    return this;
  }
  
  /**
   * Unit delta set.
   *
   * @return the delta set
   */
  public DeltaSet<K> unit() {
    return scale(1.0 / getMagnitude());
  }
  
  public StateSet<K> asState() {
    StateSet<K> returnValue = new StateSet<>();
    map.forEach((layer, delta) -> {
      returnValue.get(layer, delta.target).set(delta.delta);
    });
    return returnValue;
  }
  
  /**
   * Dot double.
   *
   * @param right the right
   * @return the double
   */
  public double dot(DoubleBufferSet<K,Delta<K>> right) {
    Stream<Map.Entry<K, Delta<K>>> stream = map.entrySet().stream();
    if(100 < map.size()) stream = stream.parallel();
    return stream.mapToDouble(entry -> {
      K key = entry.getKey();
      assert key.equals(key);
      if (right.map.containsKey(key)) {
        return entry.getValue().dot(right.map.get(key));
      }
      else {
        return 0;
      }
    }).summaryStatistics().getSum();
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
  
}

