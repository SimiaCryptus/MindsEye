package com.simiacryptus.mindseye.opencl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class MatrixMultiplyKernel extends com.amd.aparapi.Kernel {
  static final Logger log = LoggerFactory.getLogger(MatrixMultiplyKernel.class);

  private static final boolean DEBUG = false;
  double[] vector;
  double[] output;
  double[] matrix;
  int mode = 0;

  static final ResourcePool<? extends MatrixMultiplyKernel> POOL = new ResourcePool<MatrixMultiplyKernel>(16) {
    @Override
    public MatrixMultiplyKernel create() {
      final MatrixMultiplyKernel backpropTask = new MatrixMultiplyKernel();
      OpenCL.init(backpropTask);
      backpropTask.setExplicit(true);
      return backpropTask;
    }
  };

  public MatrixMultiplyKernel() {
  }

  public void exe(final com.amd.aparapi.device.Device device) {
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
    if (0 == mode) {
      for (int j = 0; j < vector.length; j++) {
        accum += matrix[j * output.length + i] * vector[j];
      } 
    } else {
      for (int j = 0; j < vector.length; j++) {
        accum += matrix[j + vector.length * i] * vector[j];
      } 
    }
    return accum;
  }

  public static void multiply(final double[] vector, final double[] matrix, final double[] output) {
    POOL.with(backpropTask -> {
      backpropTask.vector = vector;
      backpropTask.matrix = matrix;
      backpropTask.output = output;
      backpropTask.mode = 0;
      backpropTask.put(backpropTask.vector);
      backpropTask.put(backpropTask.matrix);
      OpenCL.range.with(range -> backpropTask.exe(range));
      backpropTask.get(backpropTask.output);
    });
  }
}