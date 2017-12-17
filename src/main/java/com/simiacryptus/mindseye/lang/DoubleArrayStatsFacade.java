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
import java.util.DoubleSummaryStatistics;
import java.util.stream.DoubleStream;

/**
 * The type Double array stats facade.
 */
public class DoubleArrayStatsFacade {
  /**
   * The Data.
   */
  private final double[] data;
  
  /**
   * Instantiates a new Double array stats facade.
   *
   * @param data the data
   */
  public DoubleArrayStatsFacade(final double[] data) {
    this.data = data;
  }
  
  
  /**
   * Length int.
   *
   * @return the int
   */
  public int length() {
    return data.length;
  }
  
  /**
   * Rms double.
   *
   * @return the double
   */
  public double rms() {
    return Math.sqrt(sumSq() / length());
  }
  
  /**
   * Sum double.
   *
   * @return the double
   */
  public double sum() {
    final DoubleSummaryStatistics statistics = Arrays.stream(data).summaryStatistics();
    return statistics.getSum();
  }
  
  /**
   * Sum sq double.
   *
   * @return the double
   */
  public double sumSq() {
    final DoubleStream doubleStream = Arrays.stream(data).map((final double x) -> x * x);
    final DoubleSummaryStatistics statistics = doubleStream.summaryStatistics();
    return statistics.getSum();
  }
}
