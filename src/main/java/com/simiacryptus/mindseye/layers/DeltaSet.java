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

package com.simiacryptus.mindseye.layers;

import com.simiacryptus.util.ml.Tensor;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * The type Delta set.
 */
public class DeltaSet {
  /**
   * The Map.
   */
  public final ConcurrentHashMap<NNLayer, DeltaBuffer> map = new ConcurrentHashMap<>();
  
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
  public DeltaSet(final Map<NNLayer, DeltaBuffer> collect) {
    this.map.putAll(collect);
  }
  
  /**
   * From list delta set.
   *
   * @param descent the descent
   * @return the delta set
   */
  public static DeltaSet fromList(List<DeltaBuffer> descent) {
    DeltaSet deltaSet = new DeltaSet();
    descent.forEach(buffer -> deltaSet.get(buffer.layer, buffer.target).accumulate(buffer.delta));
    return deltaSet;
  }
  
  /**
   * Get delta buffer.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the delta buffer
   */
  public DeltaBuffer get(final NNLayer layer, final double[] ptr) {
    return this.map.computeIfAbsent(layer, l -> new DeltaBuffer(ptr, layer));
  }
  
  /**
   * Get delta buffer.
   *
   * @param layer the layer
   * @param ptr   the ptr
   * @return the delta buffer
   */
  public DeltaBuffer get(final NNLayer layer, final Tensor ptr) {
    return get(layer, ptr.getData());
  }
  
  /**
   * Map delta set.
   *
   * @param mapper the mapper
   * @return the delta set
   */
  public DeltaSet map(final Function<DeltaBuffer, DeltaBuffer> mapper) {
    return new DeltaSet(this.map.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue()))));
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
  
  /**
   * Vector list.
   *
   * @return the list
   */
  public List<DeltaBuffer> vector() {
    return this.map.values().stream().filter(n -> null != n).distinct().sorted(Comparator.comparing(y -> y.getId())).collect(Collectors.toList());
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
      DeltaBuffer value = entry.getValue();
      return value.sumSq();
    }).sum();
    double sumCnt = map.entrySet().stream().mapToDouble(entry -> {
      DeltaBuffer value = entry.getValue();
      return value.length();
    }).sum();
    return Math.sqrt(sumSq / sumCnt);
  }
  
  /**
   * Dot double.
   *
   * @param right the right
   * @return the double
   */
  public double dot(DeltaSet right) {
    return map.entrySet().stream().mapToDouble(entry -> {
      if (right.map.contains(entry.getKey())) {
        return entry.getValue().dot(right.map.get(entry.getKey()));
      } else {
        return 0;
      }
    }).sum();
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
          .accumulate(buffer.delta)
          .accumulate(right.get(layer, buffer.target).delta);
    });
    return returnValue;
  }
  
  /**
   * Copy delta set.
   *
   * @return the delta set
   */
  public DeltaSet copy() {
    return map(x -> x.copy());
  }
}
