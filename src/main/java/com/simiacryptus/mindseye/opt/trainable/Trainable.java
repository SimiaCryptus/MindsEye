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

package com.simiacryptus.mindseye.opt.trainable;

import com.simiacryptus.mindseye.layers.DeltaSet;

public interface Trainable {
  Trainable.PointSample measure();
  
  default void resetToFull() {}
  
  default boolean resetSampling() {
    return false;
  }
  
  class PointSample {
    public final DeltaSet delta;
    public final DeltaSet weights;
    public final double value;
    
    public PointSample(DeltaSet delta, DeltaSet weights, double value) {
      this.delta = delta;
      this.weights = weights;
      this.value = value;
    }
    
    @Override
    public String toString() {
      final StringBuffer sb = new StringBuffer("PointSample{");
      sb.append("value=").append(value);
      sb.append('}');
      return sb.toString();
    }
  }
}
