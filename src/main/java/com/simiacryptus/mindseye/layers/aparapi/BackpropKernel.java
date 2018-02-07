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

package com.simiacryptus.mindseye.layers.aparapi;

import com.aparapi.Kernel;
import com.aparapi.device.Device;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * The type Backprop kernel.
 */
public final class BackpropKernel extends Kernel {
  
  /**
   * The Input.
   */
  public @Nullable double[] input;
  /**
   * The Input size.
   */
  public @Nullable int[] inputSize;
  /**
   * The Kernel offset.
   */
  public int[] kernelOffset;
  /**
   * The Kernel size.
   */
  public @Nullable int[] kernelSize;
  /**
   * The Output.
   */
  public @Nullable double[] output;
  /**
   * The Output size.
   */
  public @Nullable int[] outputSize;
  /**
   * The Weights.
   */
  public @Nullable double[] weights;
  
  /**
   * Instantiates a new Backprop kernel.
   */
  public BackpropKernel() {
  }
  
  /**
   * Exe.
   *
   * @param device the device
   */
  public void exe(final @NotNull Device device) {
    //assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == this.output.length;
    //assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == this.input.length;
    assert kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length;
    execute(device.createRange(input.length));
  }
  
  @Override
  public void run() {
    final int i = getGlobalId();
    input[i] = run(i);
  }
  
  /**
   * Run double.
   *
   * @param i the
   * @return the double
   */
  public final double run(final int i) {
    final int is0 = inputSize[0];
    final int is1 = is0 * inputSize[1];
    final int is2 = is1 * inputSize[2];
    final int batch = i / is2;
    final int i2 = i % is2 / is1;
    final int i1 = i % is1 / is0;
    final int i0 = i % is0;
    
    double accum = 0;
    for (int k = 0; k < weights.length; k++) {
      if (0. != weights[k]) {
        final int ks0 = kernelSize[0];
        final int ks1 = ks0 * kernelSize[1];
        final int ks2 = ks1 * kernelSize[2];
        final int k2 = k % ks2 / ks1;
        final int k1 = k % ks1 / ks0;
        final int k0 = k % ks0;
  
        final int o2 = k2 - i2 * outputSize[2];
        if (o2 >= 0 && o2 < outputSize[2]) {
          final int o1 = i1 + k1 - kernelOffset[1];
          final int o0 = i0 + k0 - kernelOffset[0];
          if (o0 < outputSize[0] && o1 < outputSize[1] && o0 >= 0 && o1 >= 0) {
            final int o = o0 + outputSize[0] * (o1 + outputSize[1] * (o2 + outputSize[2] * batch));
            accum += output[o] * weights[k];
          }
        }
      }
    }
    return accum;
  }
}