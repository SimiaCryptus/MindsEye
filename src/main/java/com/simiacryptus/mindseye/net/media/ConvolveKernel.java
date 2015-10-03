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
    assert(outputSize[0]*outputSize[1]*outputSize[2] == output.length);
  }

  @Override
  public void run() {
    output[getGlobalId()] = run(getGlobalId());
  }

  public double run(int o) {
    int o3 = o / (outputSize[0] * outputSize[1]);
    int o2 = (o % (outputSize[0]*outputSize[1])) / outputSize[0];
    int o1 = o % outputSize[0];
    
    double accum = 0;
    for(int k=0;k<weights.length;k++){
      if(0. == weights[k]) continue;
      int k3 = k / (kernelSize[0] * kernelSize[1]);
      int k2 = (k % (kernelSize[0]*kernelSize[1])) / kernelSize[0];
      int k1 = k % kernelSize[0];
      
      int i3 = k3 - inputSize[2] * o3;
      int i2 = o2-k2;
      int i1 = o1-k1;
      
      if(0 <= i1 && i1 < inputSize[0]) {
        if(0 <= i2 && i2 < inputSize[1]) {
          if(0 <= i3 && i3 < inputSize[2]) {
            int i = i1 + inputSize[0] * (i2 + inputSize[1] * i3);
            if(0. == input[i]) continue;
            accum += input[i] * weights[k];
            //System.out.println(String.format("[%s](%s) += [%s](%s) * [%s](%s) [%s,%s,%s]",o,accum,i,input[i],k,weights[k],k1,k2,k3));
            //System.out.println(String.format("k=[%s,%s,%s]  i=[%s,%s,%s]  o=[%s,%s,%s]",k1,k2,k3,i1,i2,i3,o1,o2,o3));
          }
        }
      }
    }
    return accum;
  }
  
  public void exe(com.amd.aparapi.device.Device device){
    assert(outputSize[0]*outputSize[1]*outputSize[2] == output.length);
    //for(int o=0;o<output.length;o++){ output[o] = run(o); }
    execute(device.createRange(outputSize[0]*outputSize[1]*outputSize[2]));
  }
}
