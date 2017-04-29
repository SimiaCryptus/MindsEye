package com.simiacryptus.mindseye.opencl;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.media.ConvolutionSynapseLayer;

public final class ConvolutionController {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ConvolutionController.class);
  
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
      OpenCL.devicePool.with(device -> backpropTask.exe(device));
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
      OpenCL.devicePool.with(device -> convolveTask.exe(device));
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
      OpenCL.devicePool.with(device -> kernelTask.exe(device));
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
