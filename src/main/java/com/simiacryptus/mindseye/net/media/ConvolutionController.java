package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.stream.IntStream;

import com.amd.aparapi.Range;
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
  private final ThreadedResource<Range> range;

  public ConvolutionController(int[] inputSize, int[] kernelSize) {
    this.inputSize = inputSize;
    this.kernelSize = kernelSize;
    this.outputSize = IntStream.range(0, kernelSize.length).map(i -> {
      int x;
      if (i == (kernelSize.length - 1)) {
        x = kernelSize[i] / inputSize[i];
      } else {
        x = inputSize[i] - kernelSize[i] + 1;
      }
      if(0 >= x){
        assert(false);
      }
      return x;
    }).toArray();
    assert (this.outputSize.length == 3);
    assert (this.kernelSize.length == 3);
    assert (this.inputSize.length == 3);
    double[] input = new double[this.inputSize[0] * this.inputSize[1] * this.inputSize[2]];
    double[] weights = new double[this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2]];
    double[] output = new double[this.outputSize[0] * this.outputSize[1] * this.outputSize[2]];
    
    this.convolveTask = new ThreadedResource<ConvolveKernel>() {
      @Override
      public ConvolveKernel create() {
        ConvolveKernel convolveTask = new ConvolveKernel(inputSize, input, kernelSize, weights, outputSize, output);
        convolveTask.setExecutionMode(EXECUTION_MODE.CPU);
        convolveTask.setExplicit(true);
        convolveTask.put(inputSize);
        convolveTask.put(kernelSize);
        convolveTask.put(outputSize);
        return convolveTask;
      }
    };
    this.convolveTask.with(x->{});
    
    this.kernelTask = new ThreadedResource<GradientKernel>() {
      @Override
      public GradientKernel create() {
        GradientKernel kernelTask = new GradientKernel(inputSize, input, kernelSize, weights, outputSize, output);
        kernelTask.setExecutionMode(EXECUTION_MODE.CPU);
        kernelTask.setExplicit(true);
        kernelTask.put(inputSize);
        kernelTask.put(kernelSize);
        kernelTask.put(outputSize);
        return kernelTask;
      }
    };
    this.kernelTask.with(x->{});

    this.backpropTask = new ThreadedResource<BackpropKernel>() {
      @Override
      public BackpropKernel create() {
        BackpropKernel backpropTask = new BackpropKernel(inputSize, input, kernelSize, weights, outputSize, output);
        backpropTask.setExecutionMode(EXECUTION_MODE.CPU);
        backpropTask.setExplicit(true);
        backpropTask.put(inputSize);
        backpropTask.put(kernelSize);
        backpropTask.put(outputSize);
        return backpropTask;
      }
    };
    this.backpropTask.with(x->{});
    
    this.range = new ThreadedResource<Range>() {
      @Override
      public Range create() {
        final com.amd.aparapi.device.OpenCLDevice openclDevice = (com.amd.aparapi.device.OpenCLDevice) com.amd.aparapi.device.Device.firstCPU();
        return openclDevice.createRange3D(inputSize[0], inputSize[1], inputSize[2]);
      }
    };
    this.range.with(x->{});

  }

  public synchronized void convolve(double[] input, double[] weights, double[] output) {
    convolveTask.with(convolveTask-> {
      convolveTask.input = input;
      convolveTask.weights = weights;
      convolveTask.output = output;
      convolveTask.put(convolveTask.input);
      convolveTask.put(convolveTask.weights);
      convolveTask.put(convolveTask.output);
      range.with(range->convolveTask.execute(range));
      convolveTask.get(convolveTask.output);
    });
  }

  public synchronized void gradient(double[] input, double[] output, double[] weights) {
    kernelTask.with(kernelTask-> {
      kernelTask.input = input;
      kernelTask.weights = weights;
      kernelTask.output = output;
      kernelTask.put(kernelTask.input);
      kernelTask.put(kernelTask.weights);
      kernelTask.put(kernelTask.output);
      range.with(range->kernelTask.execute(range));
      kernelTask.get(kernelTask.weights);
    });
  }

  public synchronized void backprop(double[] output, double[] weights, double[] input) {
    backpropTask.with(backpropTask-> {
      backpropTask.input = input;
      backpropTask.weights = weights;
      backpropTask.output = output;
      backpropTask.put(backpropTask.input);
      backpropTask.put(backpropTask.weights);
      backpropTask.put(backpropTask.output);
      range.with(range->backpropTask.execute(range));
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

}
