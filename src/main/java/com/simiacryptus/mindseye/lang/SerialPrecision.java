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

import jcuda.Sizeof;

import java.nio.*;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;

/**
 * This enum defines the levels of precision supported for bulk data serialization
 */
public enum SerialPrecision implements DataSerializer {
  /**
   * Double floating-point precision.
   */
  Double(Sizeof.DOUBLE) {
    @Override
    public void copy(double[] from, byte[] to) {
      DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
      DoubleBuffer outBuffer = ByteBuffer.wrap(to).asDoubleBuffer();
      while (inBuffer.hasRemaining()) {
        outBuffer.put(inBuffer.get());
      }
    }
    
    @Override
    public void copy(byte[] from, double[] to) {
      DoubleBuffer inBuffer = ByteBuffer.wrap(from).asDoubleBuffer();
      DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
      while (inBuffer.hasRemaining()) {
        outBuffer.put(inBuffer.get());
      }
    }
  },
  /**
   * Float floating-point precision.
   */
  Float(Sizeof.FLOAT) {
    @Override
    public void copy(double[] from, byte[] to) {
      DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
      FloatBuffer outBuffer = ByteBuffer.wrap(to).asFloatBuffer();
      while (inBuffer.hasRemaining()) {
        outBuffer.put((float) inBuffer.get());
      }
    
    }
  
    @Override
    public void copy(byte[] from, double[] to) {
      FloatBuffer inBuffer = ByteBuffer.wrap(from).asFloatBuffer();
      DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
      while (inBuffer.hasRemaining()) {
        outBuffer.put(inBuffer.get());
      }
    }
  },
  /**
   * 32-bit adaptive uniform precision
   */
  Uniform32(4) {
    @Override
    public void copy(double[] from, byte[] to) {
      DoubleSummaryStatistics statistics = Arrays.stream(from).summaryStatistics();
      DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
      FloatBuffer floatBuffer = ByteBuffer.wrap(to).asFloatBuffer();
      double min = statistics.getMin();
      double max = statistics.getMax();
      floatBuffer.put((float) min);
      floatBuffer.put((float) max);
      double center = (max + min) / 2;
      double radius = (max - min) / 2;
      IntBuffer byteBuffer = ByteBuffer.wrap(to).asIntBuffer();
      byteBuffer.position(2);
      while (inBuffer.hasRemaining()) {
        byteBuffer.put((int) (Integer.MAX_VALUE * (inBuffer.get() - center) / radius));
      }
      
    }
    
    @Override
    public void copy(byte[] from, double[] to) {
      DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
      FloatBuffer floatBuffer = ByteBuffer.wrap(from).asFloatBuffer();
      double min = floatBuffer.get();
      double max = floatBuffer.get();
      double center = (max + min) / 2;
      double radius = (max - min) / 2;
      IntBuffer intBuffer = ByteBuffer.wrap(from).asIntBuffer();
      intBuffer.position(2);
      while (intBuffer.hasRemaining()) {
        int v = intBuffer.get();
        outBuffer.put((v * radius / Integer.MAX_VALUE) + center);
      }
      
    }
    
    @Override
    public int getHeaderSize() {
      return 8;
    }
  },
  /**
   * 16-bit adaptive uniform precision
   */
  Uniform16(2) {
    @Override
    public void copy(double[] from, byte[] to) {
      DoubleSummaryStatistics statistics = Arrays.stream(from).summaryStatistics();
      DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
      FloatBuffer floatBuffer = ByteBuffer.wrap(to).asFloatBuffer();
      double min = statistics.getMin();
      double max = statistics.getMax();
      floatBuffer.put((float) min);
      floatBuffer.put((float) max);
      double center = (max + min) / 2;
      double radius = (max - min) / 2;
      ShortBuffer shortBuffer = ByteBuffer.wrap(to).asShortBuffer();
      shortBuffer.position(4);
      while (inBuffer.hasRemaining()) {
        shortBuffer.put((short) (Short.MAX_VALUE * (inBuffer.get() - center) / radius));
      }
      
    }
    
    @Override
    public void copy(byte[] from, double[] to) {
      DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
      FloatBuffer floatBuffer = ByteBuffer.wrap(from).asFloatBuffer();
      double min = floatBuffer.get();
      double max = floatBuffer.get();
      double center = (max + min) / 2;
      double radius = (max - min) / 2;
      ShortBuffer shortBuffer = ByteBuffer.wrap(from).asShortBuffer();
      shortBuffer.position(4);
      while (shortBuffer.hasRemaining()) {
        short v = shortBuffer.get();
        outBuffer.put((v * radius / Short.MAX_VALUE) + center);
      }
      
    }
    
    @Override
    public int getHeaderSize() {
      return 8;
    }
  },
  /**
   * 8-bit adaptive uniform precision
   */
  Uniform8(1) {
    @Override
    public void copy(double[] from, byte[] to) {
      DoubleSummaryStatistics statistics = Arrays.stream(from).summaryStatistics();
      DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
      FloatBuffer floatBuffer = ByteBuffer.wrap(to).asFloatBuffer();
      double min = statistics.getMin();
      double max = statistics.getMax();
      floatBuffer.put((float) min);
      floatBuffer.put((float) max);
      double center = (max + min) / 2;
      double radius = (max - min) / 2;
      ByteBuffer byteBuffer = ByteBuffer.wrap(to);
      byteBuffer.position(8);
      while (inBuffer.hasRemaining()) {
        byteBuffer.put((byte) (Byte.MAX_VALUE * (inBuffer.get() - center) / radius));
      }
      
    }
    
    @Override
    public void copy(byte[] from, double[] to) {
      DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
      FloatBuffer floatBuffer = ByteBuffer.wrap(from).asFloatBuffer();
      double min = floatBuffer.get();
      double max = floatBuffer.get();
      double center = (max + min) / 2;
      double radius = (max - min) / 2;
      ByteBuffer byteBuffer = ByteBuffer.wrap(from);
      byteBuffer.position(8);
      while (byteBuffer.hasRemaining()) {
        byte v = byteBuffer.get();
        outBuffer.put((v * radius / Byte.MAX_VALUE) + center);
      }
      
    }
    
    @Override
    public int getHeaderSize() {
      return 8;
    }
  };
  
  private final int size;
  
  SerialPrecision(final int size) {
    this.size = size;
  }
  
  
  /**
   * The Element Size.
   */
  @Override
  public int getElementSize() {
    return size;
  }
  
  
  /**
   * To rational rational.
   *
   * @param value     the value
   * @param maxScalar the max scalar
   * @return the rational
   */
  public Rational toRational(double value, int maxScalar) {
    Rational current = rationalRecursion(value, 0);
    for (int i = 0; i < 10; i++) {
      Rational next = rationalRecursion(value, i);
      if (next.numerator < maxScalar && next.denominator < maxScalar) {
        current = next;
      }
      else {
        break;
      }
    }
    return current;
  }
  
  private Rational rationalRecursion(double value, int recursions) {
    if (value < 0) {
      Rational rational = rationalRecursion(-value, recursions);
      return new Rational(-rational.numerator, rational.denominator);
    }
    else if (0 == value) {
      return new Rational(0, 1);
    }
    else if (value >= 1) {
      int scalar = (int) value;
      Rational rational = rationalRecursion(value - scalar, recursions);
      return new Rational(rational.numerator + (scalar * rational.denominator), rational.denominator);
    }
    else if (recursions <= 0) {
      return new Rational((int) Math.round(value), 1);
    }
    else {
      Rational rational = rationalRecursion(1.0 / value, recursions - 1);
      return new Rational(rational.denominator, rational.numerator);
    }
  }
  
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
