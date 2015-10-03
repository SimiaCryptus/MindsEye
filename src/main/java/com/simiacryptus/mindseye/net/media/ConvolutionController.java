package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.stream.IntStream;

import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device.TYPE;
import com.amd.aparapi.device.OpenCLDevice;
import com.amd.aparapi.Kernel.EXECUTION_MODE;

public final class ConvolutionController {

  // public int[] outputs;
  // public int[] script;
  private int[] inputSize;
  private int[] kernelSize;
  private int[] outputSize;
  private final ThreadedResource<? extends ConvolveKernel> convolveTask;
  private final ThreadedResource<? extends GradientKernel> kernelTask;
  private final ThreadedResource<? extends BackpropKernel> backpropTask;
  private final ThreadedResource<com.amd.aparapi.device.Device> range;

  public ConvolutionController(int[] inputSize, int[] kernelSize) {
    this.inputSize = inputSize;
    this.kernelSize = kernelSize;
    this.outputSize = ConvolutionSynapseLayer.getOutputDims(inputSize, kernelSize);
    assert (this.outputSize.length == 3);
    assert (this.kernelSize.length == 3);
    assert (this.inputSize.length == 3);
    double[] input = new double[this.inputSize[0] * this.inputSize[1] * this.inputSize[2]];
    double[] weights = new double[this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2]];
    double[] output = new double[this.outputSize[0] * this.outputSize[1] * this.outputSize[2]];

    this.convolveTask = new ThreadedResource<ConvolveKernel>() {
      @Override
      public ConvolveKernel create() {
        ConvolveKernel convolveTask = new ConvolveKernel(ConvolutionController.this.inputSize, input, ConvolutionController.this.kernelSize, weights, ConvolutionController.this.outputSize, output);
        convolveTask.setExecutionMode(getExecutionMode());
        convolveTask.addExecutionModes(getExecutionMode(), EXECUTION_MODE.SEQ);
        convolveTask.setExplicit(true);
        convolveTask.put(convolveTask.inputSize);
        convolveTask.put(convolveTask.kernelSize);
        convolveTask.put(convolveTask.outputSize);
        return convolveTask;
      }
    };
    this.convolveTask.with(x -> {
    });

    this.kernelTask = new ThreadedResource<GradientKernel>() {
      @Override
      public GradientKernel create() {
        GradientKernel kernelTask = new GradientKernel(inputSize, input, kernelSize, weights, outputSize, output);
        kernelTask.setExecutionMode(getExecutionMode());
        kernelTask.addExecutionModes(getExecutionMode(), EXECUTION_MODE.SEQ);
        kernelTask.setExplicit(true);
        kernelTask.put(kernelTask.inputSize);
        kernelTask.put(kernelTask.kernelSize);
        kernelTask.put(kernelTask.outputSize);
        return kernelTask;
      }
    };
    this.kernelTask.with(x -> {
    });

    this.backpropTask = new ThreadedResource<BackpropKernel>() {
      @Override
      public BackpropKernel create() {
        BackpropKernel backpropTask = new BackpropKernel(inputSize, input, kernelSize, weights, outputSize, output);
        backpropTask.setExecutionMode(getExecutionMode());
        backpropTask.addExecutionModes(getExecutionMode(), EXECUTION_MODE.SEQ);
        backpropTask.setExplicit(true);
        backpropTask.put(backpropTask.inputSize);
        backpropTask.put(backpropTask.kernelSize);
        backpropTask.put(backpropTask.outputSize);
        return backpropTask;
      }
    };
    this.backpropTask.with(x -> {
    });

    this.range = new ThreadedResource<com.amd.aparapi.device.Device>() {
      @Override
      public com.amd.aparapi.device.Device create() {
        com.amd.aparapi.device.Device openclDevice;
        if (getExecutionMode() == EXECUTION_MODE.CPU) {
          openclDevice = (com.amd.aparapi.device.OpenCLDevice) com.amd.aparapi.device.Device.firstCPU();
        } else if (getExecutionMode() == EXECUTION_MODE.GPU) {
          openclDevice = (com.amd.aparapi.device.OpenCLDevice) com.amd.aparapi.device.Device.bestGPU();
        } else {
          openclDevice = com.amd.aparapi.device.Device.first(TYPE.SEQ);
          if(null==openclDevice)
          {
            openclDevice = com.amd.aparapi.device.Device.firstCPU();
            openclDevice.setType(TYPE.SEQ);
          }
        }
        return openclDevice;
      }
    };
    this.range.with(x -> {
    });

  }

  public void convolve(double[] input, double[] weights, double[] output) {
    assert(outputSize[0]*outputSize[1]*outputSize[2] == output.length);
    convolveTask.with(convolveTask -> {
      convolveTask.input = input;
      convolveTask.weights = weights;
      convolveTask.output = output;
      convolveTask.put(convolveTask.input);
      convolveTask.put(convolveTask.weights);
      convolveTask.put(convolveTask.output);
      range.with(range -> convolveTask.exe(range));
      convolveTask.get(convolveTask.output);
    });
  }

  public void gradient(double[] input, double[] output, double[] weights) {
    kernelTask.with(kernelTask -> {
      kernelTask.input = input;
      kernelTask.weights = weights;
      kernelTask.output = output;
      kernelTask.put(kernelTask.input);
      kernelTask.put(kernelTask.weights);
      kernelTask.put(kernelTask.output);
      range.with(range -> kernelTask.exe(range));
      kernelTask.get(kernelTask.weights);
    });
  }

  public void backprop(double[] output, double[] weights, double[] input) {
    backpropTask.with(backpropTask -> {
      backpropTask.input = input;
      backpropTask.weights = weights;
      backpropTask.output = output;
      backpropTask.put(backpropTask.input);
      backpropTask.put(backpropTask.weights);
      backpropTask.put(backpropTask.output);
      range.with(range -> backpropTask.exe(range));
      backpropTask.get(backpropTask.input);
    });
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("Convolve [inputSize=");
    builder.append(Arrays.toString(inputSize));
    builder.append(", kernelSize=");
    builder.append(Arrays.toString(kernelSize));
    builder.append(", outputSize=");
    builder.append(Arrays.toString(outputSize));
    builder.append("]");
    return builder.toString();
  }

  public EXECUTION_MODE getExecutionMode() {
    return EXECUTION_MODE.CPU;
  }

}
