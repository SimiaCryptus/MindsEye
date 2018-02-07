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

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;

/**
 * An experimental serialization scheme which uses an encoding based on rational numbers.
 */
public class Rational32 implements DataSerializer {
  
  /**
   * Instantiates a new Rational 32.
   */
  public Rational32() {
  }
  
  /**
   * To rational serial precision . rational.
   *
   * @param value     the value
   * @param maxScalar the max scalar
   * @return the serial precision . rational
   */
  @javax.annotation.Nonnull
  public static SerialPrecision.Rational toRational(double value, int maxScalar) {
    @javax.annotation.Nonnull SerialPrecision.Rational current = continuedFractions(value, 0);
    for (int i = 0; i < 10; i++) {
      @javax.annotation.Nonnull SerialPrecision.Rational next = continuedFractions(value, i);
      if (next.numerator < maxScalar && next.denominator < maxScalar) {
        current = next;
      }
      else {
        break;
      }
    }
    return current;
  }
  
  /**
   * Rational recursion 2 serial precision . rational.
   *
   * @param value      the value
   * @param recursions the recursions
   * @return the serial precision . rational
   */
  public static SerialPrecision.Rational continuedFractions(double value, int recursions) {
    if (value < 0) {
      @javax.annotation.Nonnull SerialPrecision.Rational rational = continuedFractions(-value, recursions);
      return new SerialPrecision.Rational(-rational.numerator, rational.denominator);
    }
    else if (0 == value) {
      return new SerialPrecision.Rational(0, 1);
    }
    else if (value >= 1) {
      int scalar = (int) value;
      @javax.annotation.Nonnull SerialPrecision.Rational rational = continuedFractions(value - scalar, recursions);
      return new SerialPrecision.Rational(rational.numerator + (scalar * rational.denominator), rational.denominator);
    }
    else if (recursions <= 0) {
      return new SerialPrecision.Rational((int) Math.round(value), 1);
    }
    else {
      @javax.annotation.Nonnull SerialPrecision.Rational rational = continuedFractions(1.0 / value, recursions - 1);
      return new SerialPrecision.Rational(rational.denominator, rational.numerator);
    }
  }
  
  @Override
  public void copy(@javax.annotation.Nonnull double[] from, @javax.annotation.Nonnull byte[] to) {
    
    DoubleSummaryStatistics stat = Arrays.stream(from).summaryStatistics();
    double center = stat.getAverage();
    double radius = Math.exp(Arrays.stream(from).map(x -> x - center).map(x -> Math.log(Math.abs(x))).average().getAsDouble());
    @javax.annotation.Nonnull java.nio.DoubleBuffer inBuffer = java.nio.DoubleBuffer.wrap(from);
    @javax.annotation.Nonnull FloatBuffer floatBuffer = ByteBuffer.wrap(to).asFloatBuffer();
    floatBuffer.put((float) center);
    floatBuffer.put((float) radius);
    @javax.annotation.Nonnull ShortBuffer shortBuffer = ByteBuffer.wrap(to).asShortBuffer();
    shortBuffer.position(4);
    while (shortBuffer.hasRemaining()) {
      double v = (inBuffer.get() - center) / radius;
      @javax.annotation.Nonnull SerialPrecision.Rational rational = toRational(v, 0x7FFF);
      assert rational.denominator > 0;
      shortBuffer.put((short) (rational.numerator));
      shortBuffer.put((short) (rational.denominator));
    }
    
  }
  
  @Override
  public void copy(@javax.annotation.Nonnull byte[] from, @javax.annotation.Nonnull double[] to) {
    @javax.annotation.Nonnull java.nio.DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
    @javax.annotation.Nonnull FloatBuffer floatBuffer = ByteBuffer.wrap(from).asFloatBuffer();
    double center = floatBuffer.get();
    double radius = floatBuffer.get();
    @javax.annotation.Nonnull ShortBuffer shortBuffer = ByteBuffer.wrap(from).asShortBuffer();
    shortBuffer.position(4);
    while (shortBuffer.hasRemaining()) {
      int numerator = shortBuffer.get();
      //if(numerator < 0) numerator += Short.MAX_VALUE;
      int denominator = shortBuffer.get();
      //if(denominator < 0) denominator += Short.MAX_VALUE;
      assert denominator > 0 : String.format("%d/%d (%s)", numerator, denominator, shortBuffer);
      double v = ((numerator * 1.0 / denominator) * radius) + center;
      outBuffer.put(v);
    }
    
  }
  
  @Override
  public int getElementSize() {
    return 4;
  }
  
  @Override
  public int getHeaderSize() {
    return 8;
  }
  
  //if(next.numerator < maxNumber && next.denominator < maxNumber) return next;
  
  /**
   * The type Rational.
   */
  public static class Rational {
    /**
     * The Numerator.
     */
    public final int numerator;
    /**
     * The Denominator.
     */
    public final int denominator;
  
    /**
     * Instantiates a new Rational.
     *
     * @param numerator   the numerator
     * @param denominator the denominator
     */
    public Rational(int numerator, int denominator) {
      this.numerator = numerator;
      this.denominator = denominator;
    }
    
  }
  
}
