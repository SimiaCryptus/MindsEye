package com.simiacryptus.mindseye.opencl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class BackpropKernel extends com.aparapi.Kernel {
  static final Logger log = LoggerFactory.getLogger(BackpropKernel.class);

  private static final boolean DEBUG = false;
  double[] input;
  int[] inputSize;
  int[] kernelSize;
  double[] output;
  int[] outputSize;
  double[] weights;
  static final ResourcePool<? extends BackpropKernel> POOL = new ResourcePool<BackpropKernel>(16) {
    @Override
    public BackpropKernel create() {
      final BackpropKernel backpropTask = new BackpropKernel();
      OpenCL.init(backpropTask);
      backpropTask.setExplicit(true);
      return backpropTask;
    }
  };

  public BackpropKernel() {
  }

  public void exe(final com.aparapi.device.Device device) {
    assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == this.output.length;
    assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == this.input.length;
    assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == this.weights.length;
    if (DEBUG) {
      for (int i = 0; i < this.input.length; i++) {
        this.input[i] = run(i);
      }
    } else {
      execute(device.createRange(this.input.length));
    }
  }

  @Override
  public void run() {
    final int i = getGlobalId();
    this.input[i] = run(i);
  }

  public final double run(final int i) {
    final int is0 = this.inputSize[0];
    final int is1 = is0 * this.inputSize[1];
    final int i3 = i / is1;
    final int i2 = i % is1 / is0;
    final int i1 = i % is0;

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

      // i3 = k3 - inputSize[2] * o3;
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

      accum += this.output[o] * this.weights[k];
      if (DEBUG) {
        log.debug(String.format("[%s](%s) += [%s](%s) * [%s](%s) [%s,%s,%s]", i, accum, o, this.output[o], k, this.weights[k], k1, k2, k3));
        log.debug(String.format("k=[%s,%s,%s]  i=[%s,%s,%s]  o=[%s,%s,%s]", k1, k2, k3, i1, i2, i3, o1, o2, o3));
      }
    }
    return accum;
  }
}