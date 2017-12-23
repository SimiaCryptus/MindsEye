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

package com.simiacryptus.mindseye.eval;

import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.opt.TrainingMonitor;

/**
 * Base class for an object which can be evaluated using differential weights.
 * This represents a function without inputs and with only one output.
 * The internal weights, effectively the function's input,
 * are adjusted to minimize this output.
 */
public interface Trainable {
  /**
   * Cached cached trainable.
   *
   * @return the cached trainable
   */
  default CachedTrainable<? extends Trainable> cached() {
    return new CachedTrainable<>(this);
  }
  
  /**
   * Measure trainable . point sample.
   *
   * @param monitor the monitor
   * @return the trainable . point sample
   */
  PointSample measure(TrainingMonitor monitor);
  
  /**
   * Reset sampling boolean.
   *
   * @param seed the seed
   * @return the boolean
   */
  default boolean reseed(final long seed) {
    return false;
  }
  
}
