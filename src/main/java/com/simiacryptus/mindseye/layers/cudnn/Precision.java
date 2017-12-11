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

/**
 * The enum Precision.
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
  
  Precision(int code, int size) {
    this.code = code;
    this.size = size;
  }
  
  /**
   * Gets pointer.
   *
   * @param precision the precision
   * @param data      the data
   * @return the pointer
   */
  public static Pointer getPointer(Precision precision, float[] data) {
    switch (precision) {
      case Float: {
        return Pointer.to(data);
      }
      case Double: {
        return Pointer.to(getDoubles(data));
      }
      default:
        throw new IllegalStateException();
    }
  }
  
  /**
   * Get floats float [ ].
   *
   * @param data the data
   * @return the float [ ]
   */
  public static float[] getFloats(double[] data) {
    float[] floats = new float[data.length];
    for (int i = 0; i < data.length; i++) floats[i] = (float) data[i];
    return floats;
  }
  
  /**
   * Get doubles double [ ].
   *
   * @param data the data
   * @return the double [ ]
   */
  public static double[] getDoubles(float[] data) {
    double[] doubles = new double[data.length];
    for (int i = 0; i < data.length; i++) doubles[i] = (double) data[i];
    return doubles;
  }
  
  
  public Pointer getPointer(float... data) {
    return getPointer(this, data);
  }
  
  public Pointer getPointer(double... data) {
    return getPointer(this, data);
  }
  /**
   * Gets pointer.
   *
   * @param precision the precision
   * @param data      the data
   * @return the pointer
   */
  public static Pointer getPointer(Precision precision, double[] data) {
    switch (precision) {
      case Float: {
        return Pointer.to(getFloats(data));
      }
      case Double: {
        return Pointer.to(data);
      }
      default:
        throw new IllegalStateException();
    }
  }
  
  /**
   * Is nontrivial boolean.
   *
   * @param data the data
   * @return the boolean
   */
  public static boolean isNontrivial(float[] data) {
    for (int i = 0; i < data.length; i++) if (!java.lang.Double.isFinite(data[i])) return false;
    for (int i = 0; i < data.length; i++) if (data[i] != 0) return true;
    return false;
  }
  
  /**
   * Is nontrivial boolean.
   *
   * @param data the data
   * @return the boolean
   */
  public static boolean isNontrivial(double[] data) {
    for (int i = 0; i < data.length; i++) if (!java.lang.Double.isFinite(data[i])) return false;
    for (int i = 0; i < data.length; i++) if (data[i] != 0) return true;
    return false;
  }
}
