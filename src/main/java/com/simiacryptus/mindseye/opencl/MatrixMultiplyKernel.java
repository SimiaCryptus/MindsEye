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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.IntStream;

public final class MatrixMultiplyKernel extends Kernel {
  static final Logger log = LoggerFactory.getLogger(MatrixMultiplyKernel.class);
  static final ResourcePool<? extends MatrixMultiplyKernel> POOL = new ResourcePool<MatrixMultiplyKernel>(16) {
    @Override
    public MatrixMultiplyKernel create() {
      final MatrixMultiplyKernel kernel = new MatrixMultiplyKernel();
      kernel.setExplicit(true);
      return kernel;
    }
  };
  private static final boolean DEBUG = false;
  double[] vector;
  double[] output;
  double[] matrix;
  
  public MatrixMultiplyKernel() {
  }
  
  public static void multiply(final double[][] vector, final double[] matrix, final double[][] output) {
    int slices = 4;
    IntStream.range(0, slices).parallel().forEach(slice -> {
      POOL.with(kernel -> {
        kernel.matrix = matrix;
        kernel.put(kernel.matrix);
        for (int i = 0; i < vector.length; i++) {
          if (i % slices != slice) continue;
          kernel.vector = vector[i];
          kernel.output = output[i];
          kernel.put(kernel.vector);
          OpenCL.devicePool.with(device -> kernel.exe(device));
          kernel.get(kernel.output);
        }
      });
    });
  }
  
  public void exe(final Device device) {
    if (DEBUG) {
      for (int i = 0; i < this.output.length; i++) {
        this.output[i] = run(i);
      }
    } else {
      execute(device.createRange(this.output.length));
    }
  }
  
  @Override
  public void run() {
    final int i = getGlobalId();
    this.output[i] = run(i);
  }
  
  public final double run(final int i) {
    double accum = 0;
    for (int j = 0; j < vector.length; j++) {
      accum += matrix[j * output.length + i] * vector[j];
    }
    return accum;
  }
}