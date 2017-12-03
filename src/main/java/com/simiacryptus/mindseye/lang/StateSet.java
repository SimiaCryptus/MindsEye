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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A collection of State objects being staged for particular layers.
 * Provides indexing capabilities to reference the deltas based on physical references (to double[] objects)
 * and based on logical referants (i.e. layers)
 * Provides collection-arithmetic operations appropriate to the State's 'point' geometric archtype.
 */
public class StateSet<K> extends DoubleBufferSet<K,State<K>> {
  
  /**
   * Instantiates a new State set.
   */
  public StateSet() {
  }
  
  /**
   * Instantiates a new State set.
   *
   * @param toCopy the to copy
   */
  public StateSet(DoubleBufferSet<K,State<K>> toCopy) {
    super(toCopy);
    assert stream().allMatch(x->x instanceof State);
  }
  
  /**
   * Instantiates a new State set as a copy of the target data buffers in the input set
   *
   * @param toCopy the to copy
   */
  public StateSet(DeltaSet<K> toCopy) {
    assert (toCopy.stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    toCopy.getMap().forEach((layer, layerDelta) -> {
      this.get(layer, layerDelta.target).backup();
    });
    assert (this.stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    assert stream().allMatch(x->x instanceof State);
  }
  
  /**
   * Instantiates a new State set.
   *
   * @param collect the collect
   */
  public StateSet(Map<K, State<K>> collect) {
    super(collect);
  }
  
  @Override
  protected State factory(K layer, double[] ptr) {
    return new State(layer, ptr);
  }
  
  /**
   * Union delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DoubleBufferSet<K,State<K>> union(DoubleBufferSet<K,State<K>> right) {
    return union(this, right);
  }
  
  /**
   * Union state set.
   *
   * @param left  the left
   * @param right the right
   * @return the state set
   */
  public static <K> StateSet<K> union(DoubleBufferSet<K,State<K>> left, DoubleBufferSet<K,State<K>> right) {
    Map<K, State<K>> collect = Stream.concat(
      left.map.entrySet().stream(),
      right.map.entrySet().stream()
    ).collect(Collectors.groupingBy((Map.Entry<K, State<K>> e1) -> e1.getKey(),
      Collectors.mapping((Map.Entry<K, State<K>> x) -> x.getValue(), Collectors.collectingAndThen(
        Collectors.reducing((State<K> a, State<K> b) -> {
          assert a.target == b.target;
          assert a.layer.equals(b.layer);
          return a;
        }), x -> x.get()))));
    return new StateSet(collect);
  }
  
  
  /**
   * Subtract delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DeltaSet<K> subtract(StateSet<K> right) {
    return this.add(right.asVector().scale(-1)).asVector();
  }
  
  public DeltaSet<K> asVector() {
    HashMap<K, Delta> newMap = new HashMap<>();
    map.forEach((layer, state)->new Delta(layer, state.target, state.delta));
    return new DeltaSet(newMap);
  }
  
  /**
   * Subtract delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public StateSet<K> subtract(DeltaSet<K> right) {
    return this.add(right.scale(-1));
  }
  
  /**
   * Add delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public StateSet<K> add(DeltaSet<K> right) {
    DeltaSet<K> deltas = new DeltaSet();
    map.forEach(100, (K layer, State<K> buffer) -> {
      deltas.get(layer, buffer.target).set(buffer.getDelta());
    });
    right.map.forEach(100, (K layer, Delta<K> buffer) -> {
      deltas.get(layer, buffer.target).addInPlace(buffer.getDelta());
    });
    return deltas.asState();
  }
  
  @Override
  public StateSet<K> copy() {
    return map(x -> x.copy());
  }
  
  @Override
  public StateSet<K> map(Function<State<K>, State<K>> mapper) {
    Stream<Map.Entry<K, State<K>>> stream = map.entrySet().stream();
    if(map.size() > 100) stream = stream.parallel();
    Map<K, State<K>> newMap = stream.collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue())));
    return new StateSet<>(newMap);
  }
  
  /**
   * Is different boolean.
   *
   * @return the boolean
   */
  public boolean isDifferent() {
    return stream().parallel().anyMatch(x -> !x.areEqual());
  }
}
