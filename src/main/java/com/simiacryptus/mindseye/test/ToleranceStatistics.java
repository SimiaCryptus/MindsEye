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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.util.data.DoubleStatistics;

import java.util.stream.IntStream;

/**
 * The type Tolerance statistics.
 */
public class ToleranceStatistics {
  /**
   * The Absolute tol.
   */
  public final DoubleStatistics absoluteTol;
  /**
   * The Relative tol.
   */
  public final DoubleStatistics relativeTol;
  
  /**
   * Instantiates a new Tolerance statistics.
   */
  public ToleranceStatistics() {
    this(new DoubleStatistics(), new DoubleStatistics());
  }
  
  /**
   * Instantiates a new Tolerance statistics.
   *
   * @param absoluteTol the absolute tol
   * @param relativeTol the relative tol
   */
  public ToleranceStatistics(final DoubleStatistics absoluteTol, final DoubleStatistics relativeTol) {
    this.absoluteTol = absoluteTol;
    this.relativeTol = relativeTol;
  }
  
  /**
   * Accumulate tolerance statistics.
   *
   * @param target the target
   * @param val    the val
   * @return the tolerance statistics
   */
  public ToleranceStatistics accumulate(final double target, final double val) {
    absoluteTol.accept(Math.abs(target - val));
    if (Double.isFinite(val + target) && val != -target) {
      relativeTol.accept(Math.abs(target - val) / (Math.abs(val) + Math.abs(target)));
    }
    return this;
  }
  
  /**
   * Accumulate tolerance statistics.
   *
   * @param target the target
   * @param val    the val
   * @return the tolerance statistics
   */
  public ToleranceStatistics accumulate(final double[] target, final double[] val) {
    if (target.length != val.length) throw new IllegalArgumentException();
    IntStream.range(0, target.length).forEach(i -> accumulate(target[i], val[i]));
    return this;
  }
  
  /**
   * Combine tolerance statistics.
   *
   * @param right the right
   * @return the tolerance statistics
   */
  public ToleranceStatistics combine(final ToleranceStatistics right) {
    return new ToleranceStatistics(
      absoluteTol.combine(right.absoluteTol),
      relativeTol.combine(right.relativeTol)
    );
  }
  
  @Override
  public String toString() {
    return "ToleranceStatistics{" +
      "absoluteTol=" + absoluteTol +
      ", relativeTol=" + relativeTol +
      '}';
  }
}
