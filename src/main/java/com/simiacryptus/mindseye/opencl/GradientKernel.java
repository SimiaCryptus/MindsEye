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

package com.simiacryptus.mindseye.opencl;

import com.aparapi.Kernel;
import com.aparapi.device.Device;
import com.simiacryptus.util.lang.ResourcePool;

public final class GradientKernel extends Kernel {

  double[] input;
  int[] inputSize;
  int[] kernelSize;
  double[] output;
  int[] outputSize;
  double[] weights;
  
  public GradientKernel() {
  }
  
  public void exe(final Device device) {
    assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == this.output.length;
    assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == this.input.length;
    assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == this.weights.length;
    execute(device.createRange(this.weights.length));
  }
  
  @Override
  public void run() {
    this.weights[getGlobalId()] = run(getGlobalId());
  }
  
  public double run(final int k) {
    final int ks0 = this.kernelSize[0];
    final int ks1 = ks0 * this.kernelSize[1];
    final int k2 = k / ks1;
    final int k1 = k % ks1 / ks0;
    final int k0 = k % ks0;
    
    double accum = 0.;
    for (int i = 0; i < this.input.length; i++) {
      if (0. != this.input[i]) {
        final int is0 = this.inputSize[0];
        final int is1 = is0 * this.inputSize[1];
        final int i2 = i / is1;
        final int i1 = i % is1 / is0;
        final int i0 = i % is0;
  
        final int o2 = k2 - i2 * this.outputSize[2];
        if(o2 >= 0 && o2 < this.outputSize[2]) {
          final int o1 = i1 + k1;
          final int o0 = i0 + k0;
          if(o0 < this.outputSize[0] && o1 < this.outputSize[1]) {
            final int o = o0 + this.outputSize[0] * (o1 + this.outputSize[1] * o2);
            accum += this.input[i] * this.output[o];
          }
        }
      }
    }
    return accum;
  }
}