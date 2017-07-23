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

public final class GradientKernel extends Kernel {

  double[] input;
  int[] inputSize;
  int[] kernelSize;
  double[] output;
  int[] outputSize;
  double[] weights;
  int weightSize;
  int paralellism;
  int[] kernelOffset;
  
  public GradientKernel() {
  }
  
  public void exe(final Device device) {
    //assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == this.output.length;
    //assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == this.input.length;
    assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == this.weightSize;
    execute(device.createRange2D(weightSize, paralellism));
  }
  
  @Override
  public void run() {
    final int k = getGlobalId(0);
    final int threadNumber = getGlobalId(1);
    final int ks0 = this.kernelSize[0];
    final int ks1 = ks0 * this.kernelSize[1];
    final int k2 = k / ks1;
    final int k1 = k % ks1 / ks0;
    final int k0 = k % ks0;
  
    double accum = 0.;
    for (int i = threadNumber; i < this.input.length; i+=paralellism) {
      if (0. != this.input[i]) {
        final int is0 = this.inputSize[0];
        final int is1 = is0 * this.inputSize[1];
        final int is2 = is1 * this.inputSize[2];
        final int batch = i / is2;
        final int i2 = i % is2 / is1;
        final int i1 = i % is1 / is0;
        final int i0 = i % is0;
  
        final int o2 = k2 - i2 * this.outputSize[2];
        if(o2 >= 0 && o2 < this.outputSize[2]) {
          final int o1 = i1 + k1 - kernelOffset[1];
          final int o0 = i0 + k0 - kernelOffset[0];
          if(o0 < this.outputSize[0] && o1 < this.outputSize[1] && o0 >= 0 && o1 >= 0) {
            final int o = o0 + this.outputSize[0] * (o1 + this.outputSize[1] * (o2 + this.outputSize[2] * batch));
            accum += this.input[i] * this.output[o];
          }
        }
      }
    }
    this.weights[k + weightSize * threadNumber] = accum;
  }
  
}