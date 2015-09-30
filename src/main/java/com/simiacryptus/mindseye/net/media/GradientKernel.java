package com.simiacryptus.mindseye.net.media;

public final class GradientKernel extends com.amd.aparapi.Kernel {
  final int[] inputSize;
  double[] input;
  final int[] kernelSize;
  double[] weights;
  final int[] outputSize;
  double[] output;

  public GradientKernel(int[] inputSize, double[] input, int[] kernelSize, double[] weights, int[] outputSize, double[] output) {
    this.inputSize = inputSize;
    this.input = input;
    this.kernelSize = kernelSize;
    this.weights = weights;
    this.outputSize = outputSize;
    assert(0 < this.outputSize[2]);
    this.output = output;
  }

  @Override
  public void run() {
    int o = getGlobalId(0);
    double accum = 0;
    for(int i=0;i<input.length;i++){
      int i3 = i / (inputSize[0] * inputSize[1]);
      int i2 = (i - i3*inputSize[0] * inputSize[1]) / inputSize[1];
      int i1 = (i - (i3 * inputSize[1] + i2)*inputSize[0]);
      
      int o3 = o / (outputSize[0] * outputSize[1]);
      int o2 = (o - o3*outputSize[0] * outputSize[1]) / outputSize[1];
      int o1 = (o - (o3 * outputSize[1] + o2)*outputSize[0]);
      
      int k1 = o1-i1;
      int k2 = o2-i2;
      int k3 = i3 * outputSize[2] + o3;
      
      double in_i = input[i];
      if(0. != in_i){
        if(0 <= k1 && k1 < kernelSize[0]) {
          if(0 <= k2 && k2 < kernelSize[1]) {
            if(0 <= k3 && k3 < kernelSize[2]) {
              int k = k1 + kernelSize[0] * (k2 + kernelSize[1] * k3);
              weights[k] += in_i * output[o];
            }
          }
        }
      }
    }
  }
  
  public void exe(com.amd.aparapi.device.Device device){
    execute(device.createRange(outputSize[0]*outputSize[1]*outputSize[2]));
  }
}