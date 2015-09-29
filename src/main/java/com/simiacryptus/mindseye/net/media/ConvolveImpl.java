package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;

import com.amd.aparapi.Range;
import com.amd.aparapi.Kernel.EXECUTION_MODE;

public class ConvolveImpl {

  public static final class BackpropKernel extends com.amd.aparapi.Kernel {
    private final int[] inputSize;
    private double[] input;
    private final int[] kernelSize;
    private double[] weights;
    private final int[] outputSize;
    private double[] output;

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
      int k1 = getGlobalId(1);
      for (int k2 = 0; k2 < kernelSize[1]; k2++) {
        int o1 = i1 + k1;
        int o2 = i2 + k2;
        int o = o1 + o2 * outputSize[0];
        int i = i1 + i2 * inputSize[0];
        int k = k1 + k2 * kernelSize[0];
        input[i] += weights[k] * output[o];
      }
    }
  }

  public static final class GradientKernel extends com.amd.aparapi.Kernel {
    private final int[] inputSize;
    private double[] input;
    private final int[] kernelSize;
    private double[] weights;
    private final int[] outputSize;
    private double[] output;

    public GradientKernel(int[] inputSize, double[] input, int[] kernelSize, double[] weights, int[] outputSize, double[] output) {
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
      int k1 = getGlobalId(1);
      for (int k2 = 0; k2 < kernelSize[1]; k2++) {
        int o1 = i1 + k1;
        int o2 = i2 + k2;
        int o = o1 + o2 * outputSize[0];
        int i = i1 + i2 * inputSize[0];
        int k = k1 + k2 * kernelSize[0];
        weights[k] += input[i] * output[o];
      }
    }
  }

  public static final class ConvolveKernel extends com.amd.aparapi.Kernel {
    private final int[] inputSize;
    private double[] input;
    private final int[] kernelSize;
    private double[] weights;
    private final int[] outputSize;
    private double[] output;

    public ConvolveKernel(int[] inputSize, double[] input, int[] kernelSize, double[] weights, int[] outputSize, double[] output) {
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
      int k1 = getGlobalId(1);
      for (int k2 = 0; k2 < kernelSize[1]; k2++) {
        int o1 = i1 + k1;
        int o2 = i2 + k2;
        int o = o1 + o2 * outputSize[0];
        int i = i1 + i2 * inputSize[0];
        int k = k1 + k2 * kernelSize[0];
        output[o] += input[i] * weights[k];
      }
    }
  }

  public abstract static class ThreadedResource<T> {

    final int maxItems = 16;
    public java.util.concurrent.ArrayBlockingQueue<T> pool = new ArrayBlockingQueue<>(maxItems);
    public java.util.HashSet<T> all = new java.util.HashSet<>(maxItems);

    public abstract T create();

    public void with(java.util.function.Consumer<T> f) {
      T poll = pool.poll();
      if (null == poll) {
        if(all.size() >= maxItems) {
          try {
            poll = pool.take();
          } catch (InterruptedException e) {
            throw new java.lang.RuntimeException(e);
          }
        } else {
          poll = create();
          all.add(poll);
        } 
      }
      f.accept(poll);
      pool.add(poll);
    }
  }

  // public int[] outputs;
  // public int[] script;
  private int[] inputSize;
  private int[] kernelSize;
  private int[] outputSize;
  private ThreadedResource<ConvolveImpl.ConvolveKernel> convolveTask;
  private ThreadedResource<ConvolveImpl.GradientKernel> kernelTask;
  private ThreadedResource<ConvolveImpl.BackpropKernel> backpropTask;
  private ThreadedResource<Range> range;

  public ConvolveImpl(int[] inputSize, int[] kernelSize) {
    this.inputSize = inputSize;
    this.kernelSize = kernelSize;
    this.outputSize = ConvolutionSynapseLayer.getOutputDims(inputSize, kernelSize);
    assert (this.outputSize.length == 3);
    assert (this.kernelSize.length == 3);
    assert (this.inputSize.length == 2);
    double[] input = new double[this.inputSize[0] * this.inputSize[1]];
    double[] weights = new double[this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2]];
    double[] output = new double[this.outputSize[0] * this.outputSize[1] * this.outputSize[2]];
    this.convolveTask = new ThreadedResource<ConvolveImpl.ConvolveKernel>() {
      @Override
      public ConvolveKernel create() {
        ConvolveKernel convolveTask = new ConvolveKernel(inputSize, input, kernelSize, weights, outputSize, output);
        convolveTask.setExecutionMode(EXECUTION_MODE.GPU);
        convolveTask.setExplicit(true);
        convolveTask.put(inputSize);
        convolveTask.put(kernelSize);
        convolveTask.put(outputSize);
        return convolveTask;
      }
    };
    this.kernelTask = new ThreadedResource<ConvolveImpl.GradientKernel>() {
      @Override
      public GradientKernel create() {
        GradientKernel kernelTask = new GradientKernel(inputSize, input, kernelSize, weights, outputSize, output);
        kernelTask.setExecutionMode(EXECUTION_MODE.GPU);
        kernelTask.setExplicit(true);
        kernelTask.put(inputSize);
        kernelTask.put(kernelSize);
        kernelTask.put(outputSize);
        return kernelTask;
      }
    };
    this.backpropTask = new ThreadedResource<ConvolveImpl.BackpropKernel>() {
      @Override
      public BackpropKernel create() {
        BackpropKernel backpropTask = new BackpropKernel(inputSize, input, kernelSize, weights, outputSize, output);
        backpropTask.setExecutionMode(EXECUTION_MODE.GPU);
        backpropTask.setExplicit(true);
        backpropTask.put(inputSize);
        backpropTask.put(kernelSize);
        backpropTask.put(outputSize);
        return backpropTask;
      }
    };
    this.range = new ThreadedResource<Range>() {
      @Override
      public Range create() {
        final com.amd.aparapi.device.OpenCLDevice openclDevice = (com.amd.aparapi.device.OpenCLDevice) com.amd.aparapi.device.Device.firstCPU();
        
        return openclDevice.createRange3D(inputSize[0], inputSize[1], kernelSize[0]);
        
      }
    };

  }

  public void convolve1(double[] input, double[] weights, double[] output) {
    assert (this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == output.length);
    assert (this.inputSize[0] * this.inputSize[1] == input.length);
    assert (this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == weights.length);

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

  public void calGradient(double[] input, double[] output, double[] weights) {
    assert (this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == output.length);
    assert (this.inputSize[0] * this.inputSize[1] == input.length);
    assert (this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == weights.length);

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

  public void backprop(double[] output, double[] weights, double[] input) {
    assert (this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == output.length);
    assert (this.inputSize[0] * this.inputSize[1] == input.length);
    assert (this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == weights.length);

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
