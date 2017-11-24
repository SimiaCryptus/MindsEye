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
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Delta set.
 */
public class StateSet extends DeltaSetBase<State> {
  
  public StateSet() {
  }
  
  public StateSet(DeltaSetBase<State> toCopy) {
    super(toCopy);
  }
  
  public StateSet(DeltaSet toCopy) {
    assert (this.stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    this.getMap().forEach((layer, layerDelta) -> {
      this.get(layer, layerDelta.target).backup();
    });
    assert (this.stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
  }
  
  public StateSet(Map<NNLayer, State> collect) {
    super(collect);
  }
  
  public DeltaSetBase<State> stateBackup() {
    assert (this.stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    DeltaSetBase<State> stateBackup = new Delegate(this);
    this.getMap().forEach((layer, layerDelta) -> {
      stateBackup.get(layer, layerDelta.target).backup();
    });
    assert (stateBackup.stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite)));
    return stateBackup;
  }
  
  
  @Override
  protected State factory(NNLayer layer, double[] ptr) {
    return new State(ptr,layer);
  }
  
  /**
   * Union delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DeltaSetBase<State> union(DeltaSetBase<State> right) {
    DeltaSetBase<State> left = this;
    return union(left, right);
  }
  
  public static StateSet union(DeltaSetBase<State> left, DeltaSetBase<State> right) {
    return new StateSet(Stream.concat(
      left.map.entrySet().stream(),
      right.map.entrySet().stream()
    ).collect(Collectors.groupingBy(e1 -> e1.getKey(),
      Collectors.mapping(x -> x.getValue(), Collectors.collectingAndThen(
        Collectors.reducing((a, b) -> a), x -> x.get())))));
  }
  
  
  /**
   * Subtract delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public DeltaSet subtract(StateSet right) {
    return this.add(right.asVector().scale(-1)).asVector();
  }
  
  private DeltaSet asVector() {
    HashMap<NNLayer, Delta> newMap = new HashMap<>();
    map.forEach((layer, state)->new Delta(state.delta, layer));
    return new DeltaSet(newMap);
  }
  
  /**
   * Subtract delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public StateSet subtract(DeltaSet right) {
    return this.add(right.scale(-1));
  }
  
  /**
   * Add delta set.
   *
   * @param right the right
   * @return the delta set
   */
  public StateSet add(DeltaSet right) {
    StateSet returnValue = new StateSet(this);
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
  
}
