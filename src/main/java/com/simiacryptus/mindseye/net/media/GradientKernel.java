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
    weights[getGlobalId()] = run(getGlobalId());
  }

  public double run(int k) {
    int k3 = k / (kernelSize[0] * kernelSize[1]);
    int k2 = (k % (kernelSize[0]*kernelSize[1])) / kernelSize[0];
    int k1 = k % kernelSize[0];

    double accum = 0.;
    for(int i=0;i<input.length;i++){
      if(0. == input[i]) continue;
      
      int i3 = i / (inputSize[0] * inputSize[1]);
      int i2 = (i % (inputSize[0]*inputSize[1])) / inputSize[0];
      int i1 = i % inputSize[0];

      int o3 = (k3 - i3) / inputSize[2];
      int o2 = i2+k2;
      int o1 = i1+k1;      
      
      if(0 <= o1 && o1 < outputSize[0]) {
        if(0 <= o2 && o2 < outputSize[1]) {
          if(0 <= o3 && o3 < outputSize[2]) {
            int o = o1 + outputSize[0] * (o2 + outputSize[1] * o3);
            if(0. == output[o]) continue;
            accum += input[i] * output[o];
            //System.out.println(String.format("[%s](%s) += [%s](%s) * [%s](%s) [%s,%s,%s]",k,weights[k],o,accum,i,input[i],k1,k2,k3));
            //System.out.println(String.format("k=[%s,%s,%s]  i=[%s,%s,%s]  o=[%s,%s,%s]",k1,k2,k3,i1,i2,i3,o1,o2,o3));
          }
        }
      }
    }
    return accum;
  }
  
  public void exe(com.amd.aparapi.device.Device device){
    for(int k=0;k<weights.length;k++){ weights[k] = run(k); }
    //execute(device.createRange(kernelSize[0]*kernelSize[1]*kernelSize[2]));
  }
}