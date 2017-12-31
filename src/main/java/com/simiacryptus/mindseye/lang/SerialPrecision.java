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

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

/**
 * This enum defines the levels of precision supported for bulk data serialization
 */
public enum SerialPrecision implements DataSerializer {
  /**
   * Double precision.
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
   * Float precision.
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
}
