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

public class DeltaSet {
  public final ConcurrentHashMap<NNLayer, DeltaBuffer> map = new ConcurrentHashMap<>();
  
  public DeltaSet() {
  }
  
  public DeltaSet(final Map<NNLayer, DeltaBuffer> collect) {
    this.map.putAll(collect);
  }
  
  public static DeltaSet fromList(List<DeltaBuffer> descent) {
    DeltaSet deltaSet = new DeltaSet();
    descent.forEach(buffer -> deltaSet.get(buffer.layer, buffer.target).accumulate(buffer.delta));
    return deltaSet;
  }
  
  public DeltaBuffer get(final NNLayer layer, final double[] ptr) {
    return this.map.computeIfAbsent(layer, l -> new DeltaBuffer(ptr, layer));
  }
  
  public DeltaBuffer get(final NNLayer layer, final Tensor ptr) {
    return get(layer, ptr.getData());
  }
  
  public DeltaSet map(final Function<DeltaBuffer, DeltaBuffer> mapper) {
    return new DeltaSet(this.map.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> mapper.apply(e.getValue()))));
  }
  
  public DeltaSet scale(final double f) {
    return map(x -> x.scale(f));
  }
  
  public List<DeltaBuffer> vector() {
    return this.map.values().stream().filter(n -> null != n).distinct().sorted(Comparator.comparing(y -> y.getId())).collect(Collectors.toList());
  }
  
  public DeltaSet unit() {
    return scale(1.0 / getMagnitude());
  }
  
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
  
  public double dot(DeltaSet right) {
    return map.entrySet().stream().mapToDouble(entry -> {
      if (right.map.contains(entry.getKey())) {
        return entry.getValue().dot(right.map.get(entry.getKey()));
      } else {
        return 0;
      }
    }).sum();
  }
  
  public DeltaSet add(DeltaSet right) {
    DeltaSet returnValue = new DeltaSet();
    map.forEach((layer, buffer) -> {
      DeltaBuffer returnBuffer = returnValue.get(layer, buffer.target).accumulate(buffer.delta);
      if (right.map.contains(layer)) {
        returnBuffer.accumulate(right.map.get(layer).delta);
      }
    });
    return returnValue;
  }
  
  public DeltaSet copy() {
    return map(x -> x.copy());
  }
}
