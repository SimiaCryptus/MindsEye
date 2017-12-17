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

package com.simiacryptus.mindseye.layers.aparapi;

import com.aparapi.Kernel;
import com.aparapi.device.Device;

/**
 * The type Convolve kernel.
 */
public final class ConvolveKernel extends Kernel {
  
  /**
   * The Input.
   */
  public double[] input;
  /**
   * The Input size.
   */
  public int[] inputSize;
  /**
   * The Kernel offset.
   */
  public int[] kernelOffset;
  /**
   * The Kernel size.
   */
  public int[] kernelSize;
  /**
   * The Output.
   */
  public double[] output;
  /**
   * The Output size.
   */
  public int[] outputSize;
  /**
   * The Weights.
   */
  public double[] weights;
  
  /**
   * Instantiates a new Convolve kernel.
   */
  public ConvolveKernel() {
  }
  
  /**
   * Exe.
   *
   * @param device the device
   */
  public void exe(final Device device) {
    //assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == this.output.length;
    //assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == this.input.length;
    assert null != kernelSize;
    assert null != weights;
    assert kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length;
    execute(device.createRange(output.length));
  }
  
  @Override
  public void run() {
    final int i = getGlobalId();
    final int os0 = outputSize[0];
    final int os1 = os0 * outputSize[1];
    final int os2 = os1 * outputSize[2];
    final int batch = i / os2;
    final int o2 = i % os2 / os1;
    final int o1 = i % os1 / os0;
    final int o0 = i % os0;
    
    double accum = 0;
    for (int k = 0; k < weights.length; k++) {
      if (0. != weights[k]) {
        final int ks0 = kernelSize[0];
        final int ks1 = ks0 * kernelSize[1];
        final int ks2 = ks1 * kernelSize[2];
        final int k2 = k % ks2 / ks1;
        final int k1 = k % ks1 / ks0;
        final int k0 = k % ks0;
      
        final int x = k2 - o2;
        if (x >= 0 && 0 == x % outputSize[2]) {
          final int i2 = x / outputSize[2];
          if (i2 >= 0 && i2 < inputSize[2]) {
            final int i0 = o0 - k0 + kernelOffset[0];
            final int i1 = o1 - k1 + kernelOffset[1];
            if (i0 >= 0 && i1 >= 0 && i1 < inputSize[1] && i0 < inputSize[0]) {
              final int i11 = i0 + inputSize[0] * (i1 + inputSize[1] * (i2 + inputSize[2] * batch));
              accum += input[i11] * weights[k];
            }
          }
        }
      }
    }
    output[i] = accum;
  }
  
}