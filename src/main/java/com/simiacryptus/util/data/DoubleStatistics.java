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

package com.simiacryptus.util.data;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.stream.Collector;

/**
 * From: http://stackoverflow.com/questions/36263352/java-streams-standard-deviation
 * Author: Tunaki
 */
public class DoubleStatistics extends DoubleSummaryStatistics {
  
  /**
   * The Collector.
   */
  public static Collector<Double, DoubleStatistics, DoubleStatistics> COLLECTOR = Collector.of(
    DoubleStatistics::new,
    DoubleStatistics::accept,
    DoubleStatistics::combine,
    d -> d
  );
  
  /**
   * The Numbers.
   */
  public static Collector<Number, DoubleStatistics, DoubleStatistics> NUMBERS = Collector.of(
    DoubleStatistics::new,
    (a, n) -> a.accept(n.doubleValue()),
    DoubleStatistics::combine,
    d -> d
  );

  private double sumOfSquare = 0.0d;
  private double sumOfSquareCompensation; // Low order bits of sum
  private double simpleSumOfSquare; // Used to compute right sum for non-finite inputs
  
  /**
   * Accept double statistics.
   *
   * @param value the value
   * @return the double statistics
   */
  public DoubleStatistics accept(double[] value) {
    Arrays.stream(value).forEach(this::accept);
    return this;
  }
  
  @Override
  public synchronized void accept(double value) {
    super.accept(value);
    double squareValue = value * value;
    simpleSumOfSquare += squareValue;
    sumOfSquareWithCompensation(squareValue);
  }
  
  /**
   * Combine double statistics.
   *
   * @param other the other
   * @return the double statistics
   */
  public DoubleStatistics combine(DoubleStatistics other) {
    super.combine(other);
    simpleSumOfSquare += other.simpleSumOfSquare;
    sumOfSquareWithCompensation(other.sumOfSquare);
    sumOfSquareWithCompensation(other.sumOfSquareCompensation);
    return this;
  }

  private void sumOfSquareWithCompensation(double value) {
    double tmp = value - sumOfSquareCompensation;
    double velvel = sumOfSquare + tmp; // Little wolf of rounding error
    sumOfSquareCompensation = (velvel - sumOfSquare) - tmp;
    sumOfSquare = velvel;
  }
  
  /**
   * Gets sum of square.
   *
   * @return the sum of square
   */
  public double getSumOfSquare() {
    double tmp = sumOfSquare + sumOfSquareCompensation;
    if (Double.isNaN(tmp) && Double.isInfinite(simpleSumOfSquare)) {
      return simpleSumOfSquare;
    }
    return tmp;
  }
  
  /**
   * Gets standard deviation.
   *
   * @return the standard deviation
   */
  public final double getStandardDeviation() {
    return getCount() > 0 ? Math.sqrt((getSumOfSquare() / getCount()) - Math.pow(getAverage(), 2)) : 0.0d;
  }
  
  @Override
  public String toString() {
    return toString(1);
  }
  
  /**
   * To string string.
   *
   * @param scale the scale
   * @return the string
   */
  public String toString(double scale) {
    return String.format("%.4e +- %.4e [%.4e - %.4e] (%d#)",getAverage() * scale, getStandardDeviation() * scale, getMin() * scale, getMax() * scale, getCount());
  }
}
