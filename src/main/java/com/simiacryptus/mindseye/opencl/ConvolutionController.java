package com.simiacryptus.mindseye.opencl;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.amd.aparapi.device.Device.TYPE;
import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;

public final class ConvolutionController {
  
  static final Logger log = LoggerFactory.getLogger(ConvolutionController.class);
  
  private static final ResourcePool<com.amd.aparapi.device.Device> range = new ResourcePool<com.amd.aparapi.device.Device>(16) {
    @Override
    public com.amd.aparapi.device.Device create() {
      com.amd.aparapi.device.Device openclDevice;
      if (getExecutionMode() == EXECUTION_MODE.CPU) {
        openclDevice = com.amd.aparapi.device.Device.firstCPU();
      } else if (getExecutionMode() == EXECUTION_MODE.ACC) {
        openclDevice = com.amd.aparapi.device.Device.firstACC();
      } else if (getExecutionMode() == EXECUTION_MODE.GPU) {
        openclDevice = com.amd.aparapi.device.Device.bestGPU();
      } else {
        openclDevice = com.amd.aparapi.device.Device.first(TYPE.SEQ);
        if (null == openclDevice) {
          openclDevice = com.amd.aparapi.device.Device.firstCPU();
          openclDevice.setType(TYPE.SEQ);
        }
      }
      return openclDevice;
    }
  };

  public static EXECUTION_MODE getExecutionMode() {
    return EXECUTION_MODE.CPU;
  }

  public static Kernel init(final com.amd.aparapi.Kernel kernel) {
    kernel.setExecutionMode(EXECUTION_MODE.CPU);
    kernel.addExecutionModes(EXECUTION_MODE.CPU, EXECUTION_MODE.GPU, EXECUTION_MODE.SEQ);
    return kernel;
  }

  private int[] inputSize;
  private int[] kernelSize;
  private int[] outputSize;

  public ConvolutionController(final int[] inputSize, final int[] kernelSize) {
    this.inputSize = inputSize;
    this.kernelSize = kernelSize;
    this.outputSize = ConvolutionSynapseLayer.getOutputDims(inputSize, kernelSize);
    assert this.outputSize.length == 3;
    assert this.kernelSize.length == 3;
    assert this.inputSize.length == 3;
  }

  public void backprop(final double[] input, final double[] weights, final double[] output) {
    assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == output.length;
    assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == input.length;
    assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == weights.length;
    BackpropKernel.POOL.with(backpropTask -> {
      backpropTask.input = input;
      backpropTask.weights = weights;
      backpropTask.output = output;
      backpropTask.outputSize = this.outputSize;
      backpropTask.inputSize = this.inputSize;
      backpropTask.kernelSize = this.kernelSize;
      backpropTask.put(backpropTask.outputSize);
      backpropTask.put(backpropTask.inputSize);
      backpropTask.put(backpropTask.kernelSize);
      backpropTask.put(backpropTask.weights);
      backpropTask.put(backpropTask.output);
      range.with(range -> backpropTask.exe(range));
      backpropTask.get(backpropTask.input);
    });
  }

  public void convolve(final double[] input, final double[] weights, final double[] output) {
    assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == output.length;
    ConvolveKernel.POOL.with(convolveTask -> {
      convolveTask.input = input;
      convolveTask.weights = weights;
      convolveTask.output = output;
      convolveTask.outputSize = this.outputSize;
      convolveTask.inputSize = this.inputSize;
      convolveTask.kernelSize = this.kernelSize;
      convolveTask.put(convolveTask.outputSize);
      convolveTask.put(convolveTask.inputSize);
      convolveTask.put(convolveTask.kernelSize);
      convolveTask.put(convolveTask.input);
      convolveTask.put(convolveTask.weights);
      range.with(range -> convolveTask.exe(range));
      convolveTask.get(convolveTask.output);
    });
  }

  public void gradient(final double[] input, final double[] weights, final double[] output) {
    GradientKernel.POOL.with(kernelTask -> {
      kernelTask.input = input;
      kernelTask.weights = weights;
      kernelTask.output = output;
      kernelTask.outputSize = this.outputSize;
      kernelTask.inputSize = this.inputSize;
      kernelTask.kernelSize = this.kernelSize;
      kernelTask.put(kernelTask.outputSize);
      kernelTask.put(kernelTask.inputSize);
      kernelTask.put(kernelTask.kernelSize);
      kernelTask.put(kernelTask.input);
      kernelTask.put(kernelTask.output);
      range.with(range -> kernelTask.exe(range));
      kernelTask.get(kernelTask.weights);
    });
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append("Convolve [inputSize=");
    builder.append(Arrays.toString(this.inputSize));
    builder.append(", kernelSize=");
    builder.append(Arrays.toString(this.kernelSize));
    builder.append(", outputSize=");
    builder.append(Arrays.toString(this.outputSize));
    builder.append("]");
    return builder.toString();
  }

}
