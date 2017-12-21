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
 * This enum defines the levels or precision supported by (our support of) the CuDNN library.
 * It also provides related methods involving precision-dependant code.
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
