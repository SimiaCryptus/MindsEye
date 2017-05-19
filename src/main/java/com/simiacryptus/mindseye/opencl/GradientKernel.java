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
  static final ResourcePool<? extends GradientKernel> POOL = new ResourcePool<GradientKernel>(16) {
    @Override
    public GradientKernel create() {
      final GradientKernel kernelTask = new GradientKernel();
      kernelTask.setExplicit(true);
      return kernelTask;
    }
  };
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
    // for(int k=0;k<weights.length;k++){ weights[k] = train(k); }
    execute(device.createRange(this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2]));
  }
  
  @Override
  public void run() {
    this.weights[getGlobalId()] = run(getGlobalId());
  }
  
  public double run(final int k) {
    final int ks0 = this.kernelSize[0];
    final int ks1 = ks0 * this.kernelSize[1];
    final int k3 = k / ks1;
    final int k2 = k % ks1 / ks0;
    final int k1 = k % ks0;
    
    double accum = 0.;
    for (int i = 0; i < this.input.length; i++) {
      if (0. == this.input[i]) {
        continue;
      }
      
      final int is0 = this.inputSize[0];
      final int is1 = is0 * this.inputSize[1];
      final int i3 = i / is1;
      final int i2 = i % is1 / is0;
      final int i1 = i % is0;
      
      if (0 != (k3 - i3) % this.inputSize[2]) {
        continue;
      }
      final int o3 = (k3 - i3) / this.inputSize[2];
      if (0 > o3 || o3 >= this.outputSize[2]) {
        continue;
      }
      final int o2 = i2 - k2;
      if (0 > o2 || o2 >= this.outputSize[1]) {
        continue;
      }
      final int o1 = i1 - k1;
      if (0 > o1 || o1 >= this.outputSize[0]) {
        continue;
      }
      final int o = o1 + this.outputSize[0] * (o2 + this.outputSize[1] * o3);
      if (0. == this.output[o]) {
        continue;
      }
      
      accum += this.input[i] * this.output[o];
      // System.out.println(String.format("[%s](%s) += [%s](%s) * [%s](%s)
      // [%s,%s,%s]",k,weights[k],o,accum,i,input[i],k1,k2,k3));
      // System.out.println(String.format("k=[%s,%s,%s] i=[%s,%s,%s]
      // o=[%s,%s,%s]",k1,k2,k3,i1,i2,i3,o1,o2,o3));
    }
    return accum;
  }
}