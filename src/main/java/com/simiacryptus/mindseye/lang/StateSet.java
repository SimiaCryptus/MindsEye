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

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A collection of State objects being staged for particular layers. Provides indexing capabilities to reference the
 * deltas based on physical references (to double[] objects) and based on logical referants (i.e. layers) Provides
 * collection-arithmetic operations appropriate to the State's 'point' geometric archtype.
 *
 * @param <K> the type parameter
 */
public class StateSet<K> extends DoubleBufferSet<K, State<K>> {

  /**
   * Instantiates a new State setByCoord.
   */
  public StateSet() {
  }

  /**
   * Instantiates a new State setByCoord as a copy of the target data buffers in the input setByCoord
   *
   * @param toCopy the to copy
   */
  public StateSet(@Nonnull final DeltaSet<K> toCopy) {
    assert toCopy.stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite));
    toCopy.getMap().forEach((layer, layerDelta) -> {
      this.get(layer, layerDelta.target).backup().freeRef();
    });
    assert stream().allMatch(x -> Arrays.stream(x.getDelta()).allMatch(Double::isFinite));
    assert stream().allMatch(x -> x instanceof State);
  }

  /**
   * Instantiates a new State setByCoord.
   *
   * @param toCopy the to copy
   */
  public StateSet(@Nonnull final DoubleBufferSet<K, State<K>> toCopy) {
    super(toCopy);
    assert stream().allMatch(x -> x instanceof State);
  }

  /**
   * Instantiates a new State setByCoord.
   *
   * @param collect the collect
   */
  public StateSet(@Nonnull final Map<K, State<K>> collect) {
    super(collect);
  }

  /**
   * Union state setByCoord.
   *
   * @param <K>   the type parameter
   * @param left  the left
   * @param right the right
   * @return the state setByCoord
   */
  public static <K> StateSet<K> union(@Nonnull final DoubleBufferSet<K, State<K>> left, @Nonnull final DoubleBufferSet<K, State<K>> right) {
    final Map<K, State<K>> collect = Stream.concat(
        left.map.entrySet().stream(),
        right.map.entrySet().stream()
    ).collect(Collectors.groupingBy((@Nonnull final Map.Entry<K, State<K>> e1) -> e1.getKey(),
        Collectors.mapping((@Nonnull final Map.Entry<K, State<K>> x) -> x.getValue(), Collectors.collectingAndThen(
            Collectors.reducing((@Nonnull final State<K> a, @Nonnull final State<K> b) -> {
              assert a.target == b.target;
              assert a.key.equals(b.key);
              return a;
            }), x -> x.get()))));
    return new StateSet<K>(collect);
  }

  /**
   * Add evalInputDelta setByCoord.
   *
   * @param right the right
   * @return the evalInputDelta setByCoord
   */
  @Nonnull
  public StateSet<K> add(@Nonnull final DeltaSet<K> right) {
    @Nonnull final DeltaSet<K> deltas = new DeltaSet<K>();
    map.forEach(100, (@Nonnull final K layer, @Nonnull final State<K> buffer) -> {
      deltas.get(layer, buffer.target).set(buffer.getDelta()).freeRef();
    });
    right.map.forEach(100, (@Nonnull final K layer, @Nonnull final Delta<K> buffer) -> {
      deltas.get(layer, buffer.target).addInPlace(buffer.getDelta()).freeRef();
    });
    @Nonnull StateSet<K> kStateSet = deltas.asState();
    deltas.freeRef();
    return kStateSet;
  }

  /**
   * As vector evalInputDelta setByCoord.
   *
   * @return the evalInputDelta setByCoord
   */
  @Nonnull
  public DeltaSet<K> asVector() {
    @Nonnull final HashMap<K, Delta<K>> newMap = new HashMap<>();
    map.forEach((layer, state) -> newMap.put(layer, new Delta<K>(layer, state.target, RecycleBin.DOUBLES.copyOf(state.delta, state.delta.length))));
    @Nonnull DeltaSet<K> deltaSet = new DeltaSet<>(newMap);
    newMap.values().forEach(v -> v.freeRef());
    return deltaSet;
  }

  @Nonnull
  @Override
  public StateSet<K> copy() {
    return map(x -> x.copy());
  }

  /**
   * Backup copy state setBytes.
   *
   * @return the state setBytes
   */
  @Nonnull
  public StateSet<K> backupCopy() {
    return map(l -> l.backupCopy());
  }

  /**
   * Backup state setBytes.
   *
   * @return the state setBytes
   */
  @Nonnull
  public StateSet<K> backup() {
    Stream<Map.Entry<K, State<K>>> stream = map.entrySet().stream();
    if (map.size() > 100) {
      stream = stream.parallel();
    }
    stream.forEach(e -> e.getValue().backup());
    return this;
  }

  /**
   * Restore state setBytes.
   *
   * @return the state setBytes
   */
  @Nonnull
  public StateSet<K> restore() {
    Stream<Map.Entry<K, State<K>>> stream = map.entrySet().stream();
    if (map.size() > 100) {
      stream = stream.parallel();
    }
    stream.forEach(e -> e.getValue().restore());
    return this;
  }

  @Nonnull
  @Override
  protected State<K> factory(@Nonnull final K layer, final double[] target) {
    return new State<K>(layer, target);
  }

  /**
   * Is different boolean.
   *
   * @return the boolean
   */
  public boolean isDifferent() {
    return stream().parallel().anyMatch(x -> !x.areEqual());
  }

  @Nonnull
  @Override
  public StateSet<K> map(@Nonnull final Function<State<K>, State<K>> mapper) {
    Stream<Map.Entry<K, State<K>>> stream = map.entrySet().stream();
    if (map.size() > 100) {
      stream = stream.parallel();
    }
    final Map<K, State<K>> newMap = stream.collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue())));
    @Nonnull StateSet<K> kStateSet = new StateSet<>(newMap);
    newMap.values().forEach(x -> x.freeRef());
    return kStateSet;
  }

  /**
   * Subtract evalInputDelta setByCoord.
   *
   * @param right the right
   * @return the evalInputDelta setByCoord
   */
  @Nonnull
  public StateSet<K> subtract(@Nonnull final DeltaSet<K> right) {
    return this.add(right.scale(-1));
  }

  /**
   * Subtract evalInputDelta setByCoord.
   *
   * @param right the right
   * @return the evalInputDelta setByCoord
   */
  @Nonnull
  public DeltaSet<K> subtract(@Nonnull final StateSet<K> right) {
    @Nonnull DeltaSet<K> rvec = right.asVector();
    @Nonnull DeltaSet<K> scale = rvec.scale(-1);
    rvec.freeRef();
    @Nonnull StateSet<K> add = this.add(scale);
    scale.freeRef();
    @Nonnull DeltaSet<K> addVector = add.asVector();
    add.freeRef();
    return addVector;
  }


//  /**
//   * Union evalInputDelta setByCoord.
//   *
//   * @param right the right
//   * @return the evalInputDelta setByCoord
//   */
//  @Nonnull
//  public DoubleBufferSet<K, State<K>> union(@Nonnull final DoubleBufferSet<K, State<K>> right) {
//    return StateSet.union(this, right);
//  }
}
