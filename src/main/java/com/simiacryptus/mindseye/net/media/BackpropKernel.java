package com.simiacryptus.mindseye.net.media;

public final class BackpropKernel extends com.amd.aparapi.Kernel {
  final int[] inputSize;
  double[] input;
  final int[] kernelSize;
  double[] weights;
  final int[] outputSize;
  double[] output;

  public BackpropKernel(int[] inputSize, double[] input, int[] kernelSize, double[] weights, int[] outputSize, double[] output) {
    this.inputSize = inputSize;
    this.input = input;
    this.kernelSize = kernelSize;
    this.weights = weights;
    this.outputSize = outputSize;
    this.output = output;
  }

  @Override
  public void run() {
    int i1 = getGlobalId(0);
    int i2 = getGlobalId(1);
    int i3 = getGlobalId(2);
    for (int k1 = 0; k1 < kernelSize[0]; k1++) {
      for (int k2 = 0; k2 < kernelSize[1]; k2++) {
        for (int o3 = 0; o3 < outputSize[2]; o3++) {
          int o1 = i1 + k1;
          int o2 = i2 + k2;
          int k3 = i3 * outputSize[2] + o3;
          int o = o1 + outputSize[0] * (o2 + outputSize[1] * o3);
          int i = i1 + inputSize[0] * (i2 + inputSize[1] * i3);
          int k = k1 + kernelSize[0] * (k2 + kernelSize[1] * k3);
          input[i] += weights[k] * output[o];
        }
      }
    }
  }
 
  public void exe(com.amd.aparapi.device.Device device){
    execute(device.createRange3D(inputSize[0], inputSize[1], inputSize[2]));
  }
}