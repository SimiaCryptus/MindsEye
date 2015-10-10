package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.amd.aparapi.device.Device.TYPE;

public final class ConvolutionController {
  public static final class BackpropKernel extends com.amd.aparapi.Kernel {

    private static final boolean DEBUG = false;
    double[] input;
    int[] inputSize;
    int[] kernelSize;
    double[] output;
    int[] outputSize;
    double[] weights;

    public BackpropKernel() {
    }

    public void exe(final com.amd.aparapi.device.Device device) {
      assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == this.output.length;
      assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == this.input.length;
      assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == this.weights.length;
      if (DEBUG) {
        for (int i = 0; i < this.input.length; i++) {
          this.input[i] = run(i);
        }
      } else {
        execute(device.createRange(this.input.length));
      }
    }

    @Override
    public void run() {
      final int i = getGlobalId();
      this.input[i] = run(i);
    }

    public final double run(final int i) {
      final int is0 = this.inputSize[0];
      final int is1 = is0 * this.inputSize[1];
      final int i3 = i / is1;
      final int i2 = i % is1 / is0;
      final int i1 = i % is0;

      double accum = 0;
      for (int k = 0; k < this.weights.length; k++) {
        if (0. == this.weights[k]) {
          continue;
        }
        final int ks0 = this.kernelSize[0];
        final int ks1 = ks0 * this.kernelSize[1];
        final int k3 = k / ks1;
        final int k2 = k % ks1 / ks0;
        final int k1 = k % ks0;

        // i3 = k3 - inputSize[2] * o3;
        if (0 != (k3 - i3) % this.inputSize[2]) {
          continue;
        }
        final int o3 = (k3 - i3) / this.inputSize[2];
        if (0 > o3 || o3 >= this.outputSize[2]) {
          continue;
        }
        final int o2 = i2 - k2;
        if (0 > o2 || o2 >= this.outputSize[1]) {
          continue;
        }
        final int o1 = i1 - k1;
        if (0 > o1 || o1 >= this.outputSize[0]) {
          continue;
        }
        final int o = o1 + this.outputSize[0] * (o2 + this.outputSize[1] * o3);
        if (0. == this.output[o]) {
          continue;
        }

        accum += this.output[o] * this.weights[k];
        if (DEBUG) {
          log.debug(String.format("[%s](%s) += [%s](%s) * [%s](%s) [%s,%s,%s]", i, accum, o, this.output[o], k, this.weights[k], k1, k2, k3));
          log.debug(String.format("k=[%s,%s,%s]  i=[%s,%s,%s]  o=[%s,%s,%s]", k1, k2, k3, i1, i2, i3, o1, o2, o3));
        }
      }
      return accum;
    }
  }

  public static final class ConvolveKernel extends com.amd.aparapi.Kernel {

    private static final boolean DEBUG = false;
    double[] input;
    int[] inputSize;
    int[] kernelSize;
    double[] output;
    int[] outputSize;
    double[] weights;

    public ConvolveKernel() {
    }

    public void exe(final com.amd.aparapi.device.Device device) {
      assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == this.output.length;
      assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == this.input.length;
      assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == this.weights.length;
      if (DEBUG) {
        for (int o = 0; o < this.output.length; o++) {
          this.output[o] = run(o);
        }
      } else {
        execute(device.createRange(this.outputSize[0] * this.outputSize[1] * this.outputSize[2]));
      }
    }

    @Override
    public void run() {
      final int i = getGlobalId();
      this.output[i] = run(i);
    }

    public double run(final int o) {
      final int os0 = this.outputSize[0];
      final int os1 = os0 * this.outputSize[1];
      final int o3 = o / os1;
      final int o2 = o % os1 / os0;
      final int o1 = o % os0;

      double accum = 0;
      for (int k = 0; k < this.weights.length; k++) {
        if (0. == this.weights[k]) {
          continue;
        }
        final int ks0 = this.kernelSize[0];
        final int ks1 = ks0 * this.kernelSize[1];
        final int k3 = k / ks1;
        final int k2 = k % ks1 / ks0;
        final int k1 = k % ks0;

        final int i3 = k3 - this.inputSize[2] * o3;
        if (0 > i3 || i3 >= this.inputSize[2]) {
          continue;
        }
        final int i2 = o2 + k2;
        if (0 > i2 || i2 >= this.inputSize[1]) {
          continue;
        }
        final int i1 = o1 + k1;
        if (0 > i1 || i1 >= this.inputSize[0]) {
          continue;
        }
        final int i = i1 + this.inputSize[0] * (i2 + this.inputSize[1] * i3);
        if (0. == this.input[i]) {
          continue;
        }

        accum += this.input[i] * this.weights[k];
        if (DEBUG) {
          log.debug(String.format("[%s](%s) += [%s](%s) * [%s](%s)[%s,%s,%s]", o, accum, i, this.input[i], k, this.weights[k], k1, k2, k3));
          log.debug(String.format("k=[%s,%s,%s] i=[%s,%s,%s] o=[%s,%s,%s]", k1, k2, k3, i1, i2, i3, o1, o2, o3));
        }
      }
      return accum;
    }
  }

  public static final class GradientKernel extends com.amd.aparapi.Kernel {
    double[] input;
    int[] inputSize;
    int[] kernelSize;
    double[] output;
    int[] outputSize;
    double[] weights;

    public GradientKernel() {
    }

    public void exe(final com.amd.aparapi.device.Device device) {
      assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == this.output.length;
      assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == this.input.length;
      assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == this.weights.length;
      // for(int k=0;k<weights.length;k++){ weights[k] = run(k); }
      execute(device.createRange(this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2]));
    }

    @Override
    public void run() {
      this.weights[getGlobalId()] = run(getGlobalId());
    }

    public double run(final int k) {
      final int ks0 = this.kernelSize[0];
      final int ks1 = ks0 * this.kernelSize[1];
      final int k3 = k / ks1;
      final int k2 = k % ks1 / ks0;
      final int k1 = k % ks0;

      double accum = 0.;
      for (int i = 0; i < this.input.length; i++) {
        if (0. == this.input[i]) {
          continue;
        }

        final int is0 = this.inputSize[0];
        final int is1 = is0 * this.inputSize[1];
        final int i3 = i / is1;
        final int i2 = i % is1 / is0;
        final int i1 = i % is0;

        if (0 != (k3 - i3) % this.inputSize[2]) {
          continue;
        }
        final int o3 = (k3 - i3) / this.inputSize[2];
        if (0 > o3 || o3 >= this.outputSize[2]) {
          continue;
        }
        final int o2 = i2 - k2;
        if (0 > o2 || o2 >= this.outputSize[1]) {
          continue;
        }
        final int o1 = i1 - k1;
        if (0 > o1 || o1 >= this.outputSize[0]) {
          continue;
        }
        final int o = o1 + this.outputSize[0] * (o2 + this.outputSize[1] * o3);
        if (0. == this.output[o]) {
          continue;
        }

        accum += this.input[i] * this.output[o];
        // System.out.println(String.format("[%s](%s) += [%s](%s) * [%s](%s)
        // [%s,%s,%s]",k,weights[k],o,accum,i,input[i],k1,k2,k3));
        // System.out.println(String.format("k=[%s,%s,%s] i=[%s,%s,%s]
        // o=[%s,%s,%s]",k1,k2,k3,i1,i2,i3,o1,o2,o3));
      }
      return accum;
    }
  }

  private static final ThreadedResource<? extends BackpropKernel> backpropTask = new ThreadedResource<BackpropKernel>() {
    @Override
    public BackpropKernel create() {
      final BackpropKernel backpropTask = new BackpropKernel();
      init(backpropTask);
      backpropTask.setExplicit(true);
      return backpropTask;
    }
  };

  private static final ThreadedResource<? extends ConvolveKernel> convolveTask = new ThreadedResource<ConvolveKernel>() {
    @Override
    public ConvolveKernel create() {
      final ConvolveKernel convolveTask = new ConvolveKernel();
      init(convolveTask);
      convolveTask.setExplicit(true);
      return convolveTask;
    }
  };
  private static final ThreadedResource<? extends GradientKernel> kernelTask = new ThreadedResource<GradientKernel>() {
    @Override
    public GradientKernel create() {
      final GradientKernel kernelTask = new GradientKernel();
      init(kernelTask);
      kernelTask.setExplicit(true);
      return kernelTask;
    }
  };
  private static final Logger log = LoggerFactory.getLogger(ConvolutionController.class);
  private static final ThreadedResource<com.amd.aparapi.device.Device> range = new ThreadedResource<com.amd.aparapi.device.Device>() {
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

  // public int[] outputs;
  // public int[] script;
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
    backpropTask.with(backpropTask -> {
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
    convolveTask.with(convolveTask -> {
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
    kernelTask.with(kernelTask -> {
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
