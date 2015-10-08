package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.amd.aparapi.device.Device.TYPE;
import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;

public final class ConvolutionController {
  private static final Logger log = LoggerFactory.getLogger(ConvolutionController.class);

  public static final class BackpropKernel extends com.amd.aparapi.Kernel {

    private static final boolean DEBUG = false;
    int[] outputSize;
    int[] kernelSize;
    int[] inputSize;
    double[] output;
    double[] weights;
    double[] input;

    public BackpropKernel() {
    }

    @Override
    public void run() {
      int i = getGlobalId();
      input[i] = run(i);
    }

    public final double run(int i) {
      int is0 = inputSize[0];
      int is1 = is0 * inputSize[1];
      int i3 = i / is1;
      int i2 = (i % is1) / is0;
      int i1 = i % is0;

      double accum = 0;
      for (int k = 0; k < weights.length; k++) {
        if (0. == weights[k])
          continue;
        int ks0 = kernelSize[0];
        int ks1 = ks0 * kernelSize[1];
        int k3 = k / ks1;
        int k2 = (k % ks1) / ks0;
        int k1 = k % ks0;

        // i3 = k3 - inputSize[2] * o3;
        if (0 != ((k3 - i3) % inputSize[2]))
          continue;
        int o3 = (k3 - i3) / inputSize[2];
        if (0 > o3 || o3 >= outputSize[2])
          continue;
        int o2 = (i2 - k2);
        if (0 > o2 || o2 >= outputSize[1])
          continue;
        int o1 = (i1 - k1);
        if (0 > o1 || o1 >= outputSize[0])
          continue;
        int o = o1 + outputSize[0] * (o2 + outputSize[1] * o3);
        if (0. == output[o])
          continue;

        accum += output[o] * weights[k];
        if (DEBUG) {
          log.debug(String.format("[%s](%s) += [%s](%s) * [%s](%s) [%s,%s,%s]", i, accum, o, output[o], k, weights[k], k1, k2, k3));
          log.debug(String.format("k=[%s,%s,%s]  i=[%s,%s,%s]  o=[%s,%s,%s]", k1, k2, k3, i1, i2, i3, o1, o2, o3));
        }
      }
      return accum;
    }

    public void exe(com.amd.aparapi.device.Device device) {
      assert (outputSize[0] * outputSize[1] * outputSize[2] == output.length);
      assert (inputSize[0] * inputSize[1] * inputSize[2] == input.length);
      assert (kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length);
      if (DEBUG) {
        for (int i = 0; i < input.length; i++) {
          input[i] = run(i);
        }
      } else {
        execute(device.createRange(input.length));
      }
    }
  }

  public static final class GradientKernel extends com.amd.aparapi.Kernel {
    int[] inputSize;
    double[] input;
    int[] kernelSize;
    double[] weights;
    int[] outputSize;
    double[] output;

    public GradientKernel() {
    }

    @Override
    public void run() {
      weights[getGlobalId()] = run(getGlobalId());
    }

    public double run(int k) {
      int ks0 = kernelSize[0];
      int ks1 = ks0 * kernelSize[1];
      int k3 = k / ks1;
      int k2 = (k % ks1) / ks0;
      int k1 = k % ks0;

      double accum = 0.;
      for (int i = 0; i < input.length; i++) {
        if (0. == input[i])
          continue;

        int is0 = inputSize[0];
        int is1 = is0 * inputSize[1];
        int i3 = i / is1;
        int i2 = (i % is1) / is0;
        int i1 = i % is0;

        if (0 != ((k3 - i3) % inputSize[2]))
          continue;
        int o3 = (k3 - i3) / inputSize[2];
        if (0 > o3 || o3 >= outputSize[2])
          continue;
        int o2 = i2 - k2;
        if (0 > o2 || o2 >= outputSize[1])
          continue;
        int o1 = i1 - k1;
        if (0 > o1 || o1 >= outputSize[0])
          continue;
        int o = o1 + outputSize[0] * (o2 + outputSize[1] * o3);
        if (0. == output[o])
          continue;

        accum += input[i] * output[o];
        // System.out.println(String.format("[%s](%s) += [%s](%s) * [%s](%s)
        // [%s,%s,%s]",k,weights[k],o,accum,i,input[i],k1,k2,k3));
        // System.out.println(String.format("k=[%s,%s,%s] i=[%s,%s,%s]
        // o=[%s,%s,%s]",k1,k2,k3,i1,i2,i3,o1,o2,o3));
      }
      return accum;
    }

    public void exe(com.amd.aparapi.device.Device device) {
      assert (outputSize[0] * outputSize[1] * outputSize[2] == output.length);
      assert (inputSize[0] * inputSize[1] * inputSize[2] == input.length);
      assert (kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length);
      // for(int k=0;k<weights.length;k++){ weights[k] = run(k); }
      execute(device.createRange(kernelSize[0] * kernelSize[1] * kernelSize[2]));
    }
  }

  public static final class ConvolveKernel extends com.amd.aparapi.Kernel {

    private static final boolean DEBUG = false;
    int[] inputSize;
    int[] kernelSize;
    int[] outputSize;
    double[] input;
    double[] weights;
    double[] output;

    public ConvolveKernel() {
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
      int o2 = ((o % os1) / os0);
      int o1 = (o % os0);

      double accum = 0;
      for (int k = 0; k < weights.length; k++) {
        if (0. == weights[k])
          continue;
        int ks0 = kernelSize[0];
        int ks1 = ks0 * kernelSize[1];
        int k3 = k / ks1;
        int k2 = (k % ks1) / ks0;
        int k1 = k % ks0;

        int i3 = k3 - inputSize[2] * o3;
        if (0 > i3 || i3 >= inputSize[2])
          continue;
        int i2 = o2 + k2;
        if (0 > i2 || i2 >= inputSize[1])
          continue;
        int i1 = o1 + k1;
        if (0 > i1 || i1 >= inputSize[0])
          continue;
        int i = i1 + inputSize[0] * (i2 + inputSize[1] * i3);
        if (0. == input[i])
          continue;

        accum += input[i] * weights[k];
        if (DEBUG) {
          log.debug(String.format("[%s](%s) += [%s](%s) * [%s](%s)[%s,%s,%s]", o, accum, i, input[i], k, weights[k], k1, k2, k3));
          log.debug(String.format("k=[%s,%s,%s] i=[%s,%s,%s] o=[%s,%s,%s]", k1, k2, k3, i1, i2, i3, o1, o2, o3));
        }
      }
      return accum;
    }

    public void exe(com.amd.aparapi.device.Device device) {
      assert (outputSize[0] * outputSize[1] * outputSize[2] == output.length);
      assert (inputSize[0] * inputSize[1] * inputSize[2] == input.length);
      assert (kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length);
      if (DEBUG) {
        for (int o = 0; o < output.length; o++) {
          output[o] = run(o);
        }
      } else {
        execute(device.createRange(outputSize[0] * outputSize[1] * outputSize[2]));
      }
    }
  }

  // public int[] outputs;
  // public int[] script;
  private int[] inputSize;
  private int[] kernelSize;
  private int[] outputSize;
  private static final ThreadedResource<? extends ConvolveKernel> convolveTask = new ThreadedResource<ConvolveKernel>() {
    @Override
    public ConvolveKernel create() {
      ConvolveKernel convolveTask = new ConvolveKernel();
      init(convolveTask);
      convolveTask.setExplicit(true);
      return convolveTask;
    }
  };
  private static final ThreadedResource<? extends GradientKernel> kernelTask = new ThreadedResource<GradientKernel>() {
    @Override
    public GradientKernel create() {
      GradientKernel kernelTask = new GradientKernel();
      init(kernelTask);
      kernelTask.setExplicit(true);
      return kernelTask;
    }
  };
  private static final ThreadedResource<? extends BackpropKernel> backpropTask = new ThreadedResource<BackpropKernel>() {
    @Override
    public BackpropKernel create() {
      BackpropKernel backpropTask = new BackpropKernel();
      init(backpropTask);
      backpropTask.setExplicit(true);
      return backpropTask;
    }
  };
  private static final ThreadedResource<com.amd.aparapi.device.Device> range = new ThreadedResource<com.amd.aparapi.device.Device>() {
    @Override
    public com.amd.aparapi.device.Device create() {
      com.amd.aparapi.device.Device openclDevice;
      if (getExecutionMode() == EXECUTION_MODE.CPU) {
        openclDevice = (com.amd.aparapi.device.OpenCLDevice) com.amd.aparapi.device.Device.firstCPU();
      } else if (getExecutionMode() == EXECUTION_MODE.ACC) {
        openclDevice = (com.amd.aparapi.device.OpenCLDevice) com.amd.aparapi.device.Device.firstACC();
      } else if (getExecutionMode() == EXECUTION_MODE.GPU) {
        openclDevice = (com.amd.aparapi.device.OpenCLDevice) com.amd.aparapi.device.Device.bestGPU();
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

  public ConvolutionController(int[] inputSize, int[] kernelSize) {
    this.inputSize = inputSize;
    this.kernelSize = kernelSize;
    this.outputSize = ConvolutionSynapseLayer.getOutputDims(inputSize, kernelSize);
    assert (this.outputSize.length == 3);
    assert (this.kernelSize.length == 3);
    assert (this.inputSize.length == 3);

  }

  public void convolve(double[] input, double[] weights, double[] output) {
    assert (outputSize[0] * outputSize[1] * outputSize[2] == output.length);
    convolveTask.with(convolveTask -> {
      convolveTask.input = input;
      convolveTask.weights = weights;
      convolveTask.output = output;
      convolveTask.outputSize = outputSize;
      convolveTask.inputSize = inputSize;
      convolveTask.kernelSize = kernelSize;
      convolveTask.put(convolveTask.outputSize);
      convolveTask.put(convolveTask.inputSize);
      convolveTask.put(convolveTask.kernelSize);
      convolveTask.put(convolveTask.input);
      convolveTask.put(convolveTask.weights);
      range.with(range -> convolveTask.exe(range));
      convolveTask.get(convolveTask.output);
    });
  }

  public void gradient(double[] input, double[] weights, double[] output) {
    kernelTask.with(kernelTask -> {
      kernelTask.input = input;
      kernelTask.weights = weights;
      kernelTask.output = output;
      kernelTask.outputSize = outputSize;
      kernelTask.inputSize = inputSize;
      kernelTask.kernelSize = kernelSize;
      kernelTask.put(kernelTask.outputSize);
      kernelTask.put(kernelTask.inputSize);
      kernelTask.put(kernelTask.kernelSize);
      kernelTask.put(kernelTask.input);
      kernelTask.put(kernelTask.output);
      range.with(range -> kernelTask.exe(range));
      kernelTask.get(kernelTask.weights);
    });
  }

  public void backprop(double[] input, double[] weights, double[] output) {
    assert (outputSize[0] * outputSize[1] * outputSize[2] == output.length);
    assert (inputSize[0] * inputSize[1] * inputSize[2] == input.length);
    assert (kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length);
    backpropTask.with(backpropTask -> {
      backpropTask.input = input;
      backpropTask.weights = weights;
      backpropTask.output = output;
      backpropTask.outputSize = outputSize;
      backpropTask.inputSize = inputSize;
      backpropTask.kernelSize = kernelSize;
      backpropTask.put(backpropTask.outputSize);
      backpropTask.put(backpropTask.inputSize);
      backpropTask.put(backpropTask.kernelSize);
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

  public static EXECUTION_MODE getExecutionMode() {
    return EXECUTION_MODE.CPU;
  }

  public static Kernel init(com.amd.aparapi.Kernel kernel) {
    kernel.setExecutionMode(EXECUTION_MODE.CPU);
    kernel.addExecutionModes(EXECUTION_MODE.CPU, EXECUTION_MODE.GPU, EXECUTION_MODE.SEQ);
    return kernel;
  }

}
