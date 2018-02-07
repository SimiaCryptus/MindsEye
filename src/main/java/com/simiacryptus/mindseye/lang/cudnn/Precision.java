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

package com.simiacryptus.mindseye.lang.cudnn;

import com.simiacryptus.mindseye.lang.NNLayer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnDataType;
import org.jetbrains.annotations.NotNull;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

/**
 * This enum defines the levels or precision supported by (our support of) the GpuSystem library. It also provides
 * related methods involving precision-dependant code.
 */
public enum Precision {
  /**
   * Double precision.
   */
  Double(cudnnDataType.CUDNN_DATA_DOUBLE, Sizeof.DOUBLE),
  /**
   * Float precision.
   */
  Float(cudnnDataType.CUDNN_DATA_FLOAT, Sizeof.FLOAT);
  
  /**
   * The Code.
   */
  public final int code;
  /**
   * The Size.
   */
  public final int size;
  
  Precision(final int code, final int size) {
    this.code = code;
    this.size = size;
  }
  
  /**
   * Get doubles double [ ].
   *
   * @param data the data
   * @return the double [ ]
   */
  public static double[] getDoubles(final @NotNull float[] data) {
    return copy(data, new double[data.length]);
  }
  
  /**
   * Copy double [ ].
   *
   * @param from    the from
   * @param doubles the doubles
   * @return the double [ ]
   */
  public static double[] copy(@NotNull float[] from, double[] doubles) {
    for (int i = 0; i < from.length; i++) {
      doubles[i] = from[i];
    }
    return doubles;
  }
  
  /**
   * Get floats float [ ].
   *
   * @param data the data
   * @return the float [ ]
   */
  public static float[] getFloats(final @NotNull double[] data) {
    return copy(data, new float[data.length]);
  }
  
  /**
   * Copy float [ ].
   *
   * @param from the from
   * @param to   the to
   * @return the float [ ]
   */
  public static float[] copy(@NotNull double[] from, float[] to) {
    for (int i = 0; i < from.length; i++) {
      to[i] = (float) from[i];
    }
    return to;
  }
  
  /**
   * Copy.
   *
   * @param from      the from
   * @param to        the to
   * @param precision the precision
   */
  public static void copy(@NotNull double[] from, @NotNull byte[] to, Precision precision) {
    if (precision == Float) copyFloats(from, to);
    else if (precision == Double) copyDoubles(from, to);
    else throw new RuntimeException();
  }
  
  /**
   * Copy.
   *
   * @param from      the from
   * @param to        the to
   * @param precision the precision
   */
  public static void copy(@NotNull byte[] from, @NotNull double[] to, Precision precision) {
    if (precision == Float) copyFloats(from, to);
    else if (precision == Double) copyDoubles(from, to);
    else throw new RuntimeException();
  }
  
  /**
   * Copy doubles.
   *
   * @param from the from
   * @param to   the to
   */
  public static void copyDoubles(@NotNull double[] from, @NotNull byte[] to) {
    @NotNull DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
    @NotNull DoubleBuffer outBuffer = ByteBuffer.wrap(to).asDoubleBuffer();
    while (inBuffer.hasRemaining()) {
      outBuffer.put(inBuffer.get());
    }
  }
  
  /**
   * Copy doubles.
   *
   * @param from the from
   * @param to   the to
   */
  public static void copyDoubles(@NotNull byte[] from, @NotNull double[] to) {
    @NotNull DoubleBuffer inBuffer = ByteBuffer.wrap(from).asDoubleBuffer();
    @NotNull DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
    while (inBuffer.hasRemaining()) {
      outBuffer.put(inBuffer.get());
    }
  }
  
  /**
   * Copy floats.
   *
   * @param from the from
   * @param to   the to
   */
  public static void copyFloats(@NotNull double[] from, @NotNull byte[] to) {
    @NotNull DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
    @NotNull FloatBuffer outBuffer = ByteBuffer.wrap(to).asFloatBuffer();
    while (inBuffer.hasRemaining()) {
      outBuffer.put((float) inBuffer.get());
    }
  }
  
  /**
   * Copy floats.
   *
   * @param from the from
   * @param to   the to
   */
  public static void copyFloats(@NotNull byte[] from, @NotNull double[] to) {
    @NotNull FloatBuffer inBuffer = ByteBuffer.wrap(from).asFloatBuffer();
    @NotNull DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
    while (inBuffer.hasRemaining()) {
      outBuffer.put(inBuffer.get());
    }
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public @NotNull NNLayer getCompatibilityLayer() {
    throw new RuntimeException("Not Implemented");
  }
  
  /**
   * Gets pointer.
   *
   * @param data the data
   * @return the pointer
   */
  public Pointer getPointer(final @NotNull double... data) {
    switch (this) {
      case Float:
        return Pointer.to(Precision.getFloats(data));
      case Double:
        return Pointer.to(data);
      default:
        throw new IllegalStateException();
    }
  }
  
  /**
   * Gets pointer.
   *
   * @param data the data
   * @return the pointer
   */
  public Pointer getPointer(final @NotNull float... data) {
    switch (this) {
      case Float:
        return Pointer.to(data);
      case Double:
        return Pointer.to(Precision.getDoubles(data));
      default:
        throw new IllegalStateException();
    }
  }
  
}
