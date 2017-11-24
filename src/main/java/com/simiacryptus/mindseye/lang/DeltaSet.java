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

/**
 * The type Delta set.
 */
public class DeltaSet extends DeltaSetBase<Delta> {
  
  public DeltaSet() {
  }
  
  public DeltaSet(DeltaSetBase<Delta> toCopy) {
    super(toCopy);
  }
  
  public DeltaSet(Map<NNLayer, ? extends Delta> collect) {
    super(collect);
  }
  
  @Override
  protected Delta factory(NNLayer layer, double[] ptr) {
    return new Delta(ptr,layer);
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
  public DeltaSet add(DeltaSet right) {
    DeltaSet returnValue = new DeltaSet();
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
  
  public StateSet add(StateSet right) {
    return right.add(this);
  }
  
  /**
   * Scale delta set.
   *
   * @param f the f
   * @return the delta set
   */
  public DeltaSet scale(final double f) {
    return new DeltaSet(map(x -> x.scale(f)));
  }
  
  @Override
  public DeltaSet map(Function<Delta, Delta> mapper) {
    return new DeltaSet(super.map(mapper));
  }
  
  @Override
  public DeltaSet copy() {
    return new DeltaSet(this);
  }
  
  /**
   * Accumulate delta set.
   *
   * @param alpha the alpha
   * @return the delta set
   */
  public DeltaSet accumulate(double alpha) {
    stream().forEach(d -> d.accumulate(alpha));
    return this;
  }
  
  /**
   * Accumulate delta set.
   *
   * @return the delta set
   */
  public DeltaSet accumulate() {
    accumulate(1);
    return this;
  }
  
  /**
   * Unit delta set.
   *
   * @return the delta set
   */
  public DeltaSet unit() {
    return scale(1.0 / getMagnitude());
  }
  
}

