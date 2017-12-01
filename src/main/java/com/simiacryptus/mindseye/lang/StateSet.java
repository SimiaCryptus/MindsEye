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
import java.util.stream.Collector;
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
   * Instantiates a new State set.
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
  
  /**
   * State backup double buffer set.
   *
   * @return the double buffer set
   */
  public DoubleBufferSet<K,State<K>> stateBackup() {
    assert (this.stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    DoubleBufferSet<K,State<K>> stateBackup = new Delegate(this);
    this.getMap().forEach((layer, layerDelta) -> {
      stateBackup.get(layer, layerDelta.target).backup();
    });
    assert (stateBackup.stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    return stateBackup;
  }
  
  
  @Override
  protected State factory(K layer, double[] ptr) {
    return new State(ptr,layer);
  }
  
  /**
   * Union delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DoubleBufferSet<K,State<K>> union(DoubleBufferSet<K,State<K>> right) {
    DoubleBufferSet<K,State<K>> left = this;
    return union(left, right);
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
        Collectors.reducing((State<K> a, State<K> b) -> a), x -> x.get()))));
    return new StateSet(collect);
  }
  
  
  /**
   * Subtract delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DeltaSet<K> subtract(StateSet right) {
    return this.add(right.asVector().scale(-1));
  }
  
  private DeltaSet asVector() {
    HashMap<K, Delta> newMap = new HashMap<>();
    map.forEach((layer, state)->new Delta(state.delta, layer));
    return new DeltaSet(newMap);
  }
  
  /**
   * Subtract delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DeltaSet<K> subtract(DeltaSet<K> right) {
    return this.add(right.scale(-1));
  }
  
  /**
   * Add delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DeltaSet<K> add(DeltaSet<K> right) {
    DeltaSet<K> returnValue = new DeltaSet(this);
    map.forEach(100, (layer, buffer) -> {
      returnValue.get(layer, buffer.target)
        .accumulate(buffer.getDelta());
    });
    right.map.forEach(100, (layer, buffer) -> {
      returnValue.get(layer, buffer.target)
        .accumulate(buffer.getDelta());
    });
    return returnValue;
  }
  
  @Override
  public DoubleBufferSet<K,State<K>> map(Function<State<K>, State<K>> mapper) {
    return new StateSet(super.map(mapper));
  }
  
  @Override
  public DoubleBufferSet<K,State<K>> copy() {
    return new StateSet(super.copy());
  }
}
