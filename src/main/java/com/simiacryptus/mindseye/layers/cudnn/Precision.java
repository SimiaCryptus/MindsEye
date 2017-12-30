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

package com.simiacryptus.mindseye.layers.cudnn;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnDataType;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

/**
 * This enum defines the levels or precision supported by (our support of) the CuDNN library. It also provides related
 * methods involving precision-dependant code.
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
  public static double[] getDoubles(final float[] data) {
    final double[] doubles = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      doubles[i] = data[i];
    }
    return doubles;
  }
  
  /**
   * Get floats float [ ].
   *
   * @param data the data
   * @return the float [ ]
   */
  public static float[] getFloats(final double[] data) {
    final float[] floats = new float[data.length];
    for (int i = 0; i < data.length; i++) {
      floats[i] = (float) data[i];
    }
    return floats;
  }
  
  /**
   * Copy.
   *
   * @param from      the from
   * @param to        the to
   * @param precision the precision
   */
  public static void copy(double[] from, byte[] to, Precision precision) {
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
  public static void copy(byte[] from, double[] to, Precision precision) {
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
  public static void copyDoubles(double[] from, byte[] to) {
    DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
    DoubleBuffer outBuffer = ByteBuffer.wrap(to).asDoubleBuffer();
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
  public static void copyDoubles(byte[] from, double[] to) {
    DoubleBuffer inBuffer = ByteBuffer.wrap(from).asDoubleBuffer();
    DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
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
  public static void copyFloats(double[] from, byte[] to) {
    DoubleBuffer inBuffer = DoubleBuffer.wrap(from);
    FloatBuffer outBuffer = ByteBuffer.wrap(to).asFloatBuffer();
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
  public static void copyFloats(byte[] from, double[] to) {
    FloatBuffer inBuffer = ByteBuffer.wrap(from).asFloatBuffer();
    DoubleBuffer outBuffer = DoubleBuffer.wrap(to);
    while (inBuffer.hasRemaining()) {
      outBuffer.put(inBuffer.get());
    }
  }
  
  /**
   * Gets pointer.
   *
   * @param data the data
   * @return the pointer
   */
  public Pointer getPointer(final double... data) {
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
  public Pointer getPointer(final float... data) {
    switch (this) {
      case Float:
        return Pointer.to(data);
      case Double:
        return Pointer.to(Precision.getDoubles(data));
      default:
        throw new IllegalStateException();
    }
  }
  
  
  /**
   * Java ptr cuda ptr.
   *
   * @param deviceNumber the device number
   * @param data         the data
   * @return the cuda ptr
   */
  public CudaPtr javaPtr(final int deviceNumber, final double... data) {
    return new CudaPtr(getPointer(data), data.length * size, deviceNumber);
  }
  
}
