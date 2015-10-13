package com.simiacryptus.mindseye.opencl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ConvolveKernel extends com.amd.aparapi.Kernel {
  static final Logger log = LoggerFactory.getLogger(ConvolveKernel.class);

  private static final boolean DEBUG = false;
  double[] input;
  int[] inputSize;
  int[] kernelSize;
  double[] output;
  int[] outputSize;
  double[] weights;
  static final ResourcePool<? extends ConvolveKernel> POOL = new ResourcePool<ConvolveKernel>(16) {
    @Override
    public ConvolveKernel create() {
      final ConvolveKernel convolveTask = new ConvolveKernel();
      OpenCL.init(convolveTask);
      convolveTask.setExplicit(true);
      return convolveTask;
    }
  };

  public ConvolveKernel() {
  }

  public void exe(final com.amd.aparapi.device.Device device) {
    assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == this.output.length;
    assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == this.input.length;
    assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == this.weights.length;
    if (DEBUG) {
      for (int o = 0; o < this.output.length; o++) {
        this.output[o] = run(o);
      }
    } else {
      execute(device.createRange(this.outputSize[0] * this.outputSize[1] * this.outputSize[2]));
    }
  }

  @Override
  public void run() {
    final int i = getGlobalId();
    this.output[i] = run(i);
  }

  public double run(final int o) {
    final int os0 = this.outputSize[0];
    final int os1 = os0 * this.outputSize[1];
    final int o3 = o / os1;
    final int o2 = o % os1 / os0;
    final int o1 = o % os0;

    double accum = 0;
    for (int k = 0; k < this.weights.length; k++) {
      if (0. == this.weights[k]) {
        continue;
      }
      final int ks0 = this.kernelSize[0];
      final int ks1 = ks0 * this.kernelSize[1];
      final int k3 = k / ks1;
      final int k2 = k % ks1 / ks0;
      final int k1 = k % ks0;

      final int i3 = k3 - this.inputSize[2] * o3;
      if (0 > i3 || i3 >= this.inputSize[2]) {
        continue;
      }
      final int i2 = o2 + k2;
      if (0 > i2 || i2 >= this.inputSize[1]) {
        continue;
      }
      final int i1 = o1 + k1;
      if (0 > i1 || i1 >= this.inputSize[0]) {
        continue;
      }
      final int i = i1 + this.inputSize[0] * (i2 + this.inputSize[1] * i3);
      if (0. == this.input[i]) {
        continue;
      }

      accum += this.input[i] * this.weights[k];
      if (DEBUG) {
        log.debug(String.format("[%s](%s) += [%s](%s) * [%s](%s)[%s,%s,%s]", o, accum, i, this.input[i], k, this.weights[k], k1, k2, k3));
        log.debug(String.format("k=[%s,%s,%s] i=[%s,%s,%s] o=[%s,%s,%s]", k1, k2, k3, i1, i2, i3, o1, o2, o3));
      }
    }
    return accum;
  }
}