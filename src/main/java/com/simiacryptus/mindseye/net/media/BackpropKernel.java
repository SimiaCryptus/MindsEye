package com.simiacryptus.mindseye.net.media;

public final class BackpropKernel extends com.amd.aparapi.Kernel {
  final int[] outputSize;
  final int[] kernelSize;
  final int[] inputSize;
  double[] output;
  double[] weights;
  double[] input;

  public BackpropKernel(int[] outputSize, double[] output, int[] kernelSize, double[] weights, int[] inputSize, double[] input) {
    this.outputSize = outputSize;
    this.output = output;
    this.kernelSize = kernelSize;
    this.weights = weights;
    this.inputSize = inputSize;
    this.input = input;
  }

  @Override
  public void run() {
    int o = getGlobalId();
    input[o] = run(o);
  }

  public double run(int i) {
    int i3 = i / (inputSize[0] * inputSize[1]);
    int i2 = (i - i3*inputSize[0] * inputSize[1]) / inputSize[1];
    int i1 = (i - (i3 * inputSize[1] + i2)*inputSize[0]);
    
    double accum = 0;
    for(int k=0;k<weights.length;k++){
      if(0. == weights[k]) continue;
      int k3 = k / (kernelSize[0] * kernelSize[1]);
      int k2 = (k - k3*kernelSize[0] * kernelSize[1]) / kernelSize[1];
      int k1 = (k - (k3 * kernelSize[1] + k2)*kernelSize[0]);
      
      int o3 = (k3 - i3) / inputSize[2];
      int o2 = i2+k2;
      int o1 = i1+k1;
      
      if(0 <= o1 && o1 < outputSize[0]) {
        if(0 <= o2 && o2 < outputSize[1]) {
          if(0 <= o3 && o3 < outputSize[2]) {
            int o = o1 + outputSize[0] * (o2 + outputSize[1] * o3);
            if(0. == output[o]) continue;;
            accum += output[o] * weights[k];
            //System.in.println(Stroutg.format("[%s](%s) += [%s](%s) * [%s](%s) [%s,%s,%s]",o,accum,i,output[i],k,weights[k],k1,k2,k3));
            //System.in.println(Stroutg.format("k=[%s,%s,%s]  i=[%s,%s,%s]  o=[%s,%s,%s]",k1,k2,k3,i1,i2,i3,o1,o2,o3));
          }
        }
      }
    }
    return accum;
  }
  
  public void exe(com.amd.aparapi.device.Device device){
    //for(int i=0;i<input.length;i++){ input[i] = run(i); }
    execute(device.createRange(inputSize[0]*inputSize[1]*inputSize[2]));
  }
}