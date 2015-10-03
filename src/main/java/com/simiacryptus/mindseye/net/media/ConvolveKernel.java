package com.simiacryptus.mindseye.net.media;

public final class ConvolveKernel extends com.amd.aparapi.Kernel {
  final int[] inputSize;
  final int[] kernelSize;
  final int[] outputSize;
  double[] input;
  double[] weights;
  double[] output;

  public ConvolveKernel(int[] inputSize, double[] input, int[] kernelSize, double[] weights, int[] outputSize, double[] output) {
    this.inputSize = inputSize;
    this.input = input;
    this.kernelSize = kernelSize;
    this.weights = weights;
    this.outputSize = outputSize;
    this.output = output;
    assert (outputSize[0] * outputSize[1] * outputSize[2] == output.length);
  }

  @Override
  public void run() {
    int i = getGlobalId();
    output[i] = run(i);
  }

  public double run(int o) {
    int os0 = outputSize[0];
    int os1 = os0 * outputSize[1];
    int o3 = (o / os1);
    int o2 = ((o % os1) / os0) + ((kernelSize[1]/2)+1);
    int o1 = (o % os0) + ((kernelSize[0]/2)+1);

    double accum = 0;
    for (int k = 0; k < weights.length; k++) {
      if (0. == weights[k]) continue;
      int ks0 = kernelSize[0];
      int ks1 = ks0 * kernelSize[1];
      int k3 = k / ks1;
      int k2 = (k % ks1) / ks0;
      int k1 = k % ks0;

      int i3 = k3 - inputSize[2] * o3;
      if (0 > i3 || i3 >= inputSize[2]) continue;
      int i2 = o2 - k2;
      if (0 > i2 || i2 >= inputSize[1]) continue;
      int i1 = o1 - k1;
      if (0 > i1 || i1 >= inputSize[0]) continue;
      int i = i1 + inputSize[0] * (i2 + inputSize[1] * i3);
      if (0. == input[i]) continue;
      
      accum += input[i] * weights[k];
      // System.out.println(String.format("[%s](%s) += [%s](%s) * [%s](%s)
      // [%s,%s,%s]",o,accum,i,input[i],k,weights[k],k1,k2,k3));
      // System.out.println(String.format("k=[%s,%s,%s] i=[%s,%s,%s]
      // o=[%s,%s,%s]",k1,k2,k3,i1,i2,i3,o1,o2,o3));
    }
    return accum;
  }

  public void exe(com.amd.aparapi.device.Device device) {
    assert (outputSize[0] * outputSize[1] * outputSize[2] == output.length);
    assert (inputSize[0] * inputSize[1] * inputSize[2] == input.length);
    assert (kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length);
    // for(int o=0;o<output.length;o++){ output[o] = run(o); }
    execute(device.createRange(outputSize[0] * outputSize[1] * outputSize[2]));
  }
}
