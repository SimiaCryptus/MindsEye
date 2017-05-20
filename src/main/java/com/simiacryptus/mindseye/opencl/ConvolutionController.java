/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.opencl;

import com.aparapi.Kernel;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.simiacryptus.mindseye.net.media.ImgConvolutionSynapseLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public final class ConvolutionController {
  
  private final BackpropKernel backpropTask = new BackpropKernel();
  private final ConvolveKernel convolveTask = new ConvolveKernel();
  private final GradientKernel kernelTask = new GradientKernel();
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ConvolutionController.class);
  
  private final int[] inputSize;
  private final int[] kernelSize;
  private final int[] outputSize;
  
  public ConvolutionController(final int[] inputSize, final int[] kernelSize) {
    this.inputSize = inputSize;
    this.kernelSize = kernelSize;
    this.outputSize = ImgConvolutionSynapseLayer.getOutputDims(inputSize, kernelSize);
    assert this.outputSize.length == 3;
    assert this.kernelSize.length == 3;
    assert this.inputSize.length == 3;
  }
  
  public void backprop(final double[][] input, final double[] weights, final double[][] output) {
    assert(input.length == output.length);
    int inLength = input[0].length;
    int outLength = output[0].length;
    double[] inputBuffer = new double[inLength * input.length];
    double[] outputBuffer = new double[outLength * output.length];
    for (int i=0;i<input.length;i++) {
      assert outLength == output[i].length;
      System.arraycopy(output[i], 0, outputBuffer, i * outLength, outLength);
    }
    backprop(inputBuffer,weights,outputBuffer);
    for (int i=0;i<input.length;i++) {
      assert inLength == input[i].length;
      System.arraycopy(inputBuffer, i * inLength, input[i], 0, inLength);
    }

  }
  
  public void convolve(final double[][] input, final double[] weights, final double[][] output) {
    assert(input.length == output.length);
    int inLength = input[0].length;
    int outLength = output[0].length;
    double[] inputBuffer = new double[inLength * input.length];
    double[] outputBuffer = new double[outLength * output.length];
    for (int i=0;i<input.length;i++) {
      assert inLength == input[i].length;
      System.arraycopy(input[i], 0, inputBuffer, i * inLength, inLength);
    }
    convolve(inputBuffer,weights,outputBuffer);
    for (int i=0;i<input.length;i++) {
      assert outLength == output[i].length;
      System.arraycopy(outputBuffer, i * outLength, output[i], 0, outLength);
    }
  }
  
  public void gradient(final double[][] input, final double[] weights, final double[][] output) {
    assert(input.length == output.length);
    int inLength = input[0].length;
    int outLength = output[0].length;
    double[] inputBuffer = new double[inLength * input.length];
    double[] outputBuffer = new double[outLength * output.length];
    for (int i=0;i<input.length;i++) {
      assert inLength == input[i].length;
      assert outLength == output[i].length;
      System.arraycopy(input[i], 0, inputBuffer, i * inLength, inLength);
      System.arraycopy(output[i], 0, outputBuffer, i * outLength, outLength);
    }
    gradient(inputBuffer,weights,outputBuffer);
  }
  
  public void backprop(final double[] input, final double[] weights, final double[] output) {
    assert this.outputSize[0] * this.outputSize[1] * this.outputSize[2] == output.length;
    assert this.inputSize[0] * this.inputSize[1] * this.inputSize[2] == input.length;
    assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == weights.length;
    OpenCL.devicePool.with(device -> {
      try {
        System.err.println("Backprop " + this);
        backpropTask.input = input;
        backpropTask.weights = weights;
        backpropTask.output = output;
        backpropTask.outputSize = this.outputSize;
        backpropTask.inputSize = this.inputSize;
        backpropTask.kernelSize = this.kernelSize;
        before();
        backpropTask.setExplicit(true);
        backpropTask.put(backpropTask.outputSize);
        backpropTask.put(backpropTask.inputSize);
        backpropTask.put(backpropTask.kernelSize);
        backpropTask.put(backpropTask.weights);
        backpropTask.put(backpropTask.output);
//        Thread.sleep(1);
        backpropTask.exe(device);
        join(backpropTask);
        backpropTask.get(backpropTask.input);
        backpropTask.cleanUpArrays();
      } catch (Throwable e) {
        throw new RuntimeException("Error with " +this,e);
      }
    });
  }
  
  private void join(Kernel kernel) {
//    try {
//      while (kernel.isExecuting()) Thread.sleep(0);
//      Thread.sleep(100);
//    } catch (InterruptedException e) {
//      Thread.currentThread().interrupt();
//    }
  }
  
  public void convolve(final double[] input, final double[] weights, final double[] output) {
    OpenCL.devicePool.with(device -> {
      try {
        System.err.println("Convolve " + this);
        convolveTask.input = input;
        convolveTask.weights = weights;
        convolveTask.output = output;
        convolveTask.outputSize = this.outputSize;
        convolveTask.inputSize = this.inputSize;
        convolveTask.kernelSize = this.kernelSize;
        before();
        convolveTask.setExplicit(true);
        convolveTask.put(convolveTask.outputSize);
        convolveTask.put(convolveTask.inputSize);
        convolveTask.put(convolveTask.kernelSize);
        convolveTask.put(convolveTask.input);
        convolveTask.put(convolveTask.weights);
//        Thread.sleep(1);
        if(device.getType() == Device.TYPE.GPU) {
        
        }
        convolveTask.exe(device);
        join(convolveTask);
        convolveTask.get(convolveTask.output);
        convolveTask.cleanUpArrays();
      } catch (Throwable e) {
        throw new RuntimeException("Error with " +this,e);
      }
    });
  }
  
  public void gradient(final double[] input, final double[] weights, final double[] output) {
    OpenCL.devicePool.with(device -> {
      try {
        System.err.println("Gradient " + this);
        kernelTask.input = input;
        kernelTask.weights = weights;
        kernelTask.output = output;
        kernelTask.outputSize = this.outputSize;
        kernelTask.inputSize = this.inputSize;
        kernelTask.kernelSize = this.kernelSize;
        before();
        kernelTask.setExplicit(true);
        kernelTask.put(kernelTask.outputSize);
        kernelTask.put(kernelTask.inputSize);
        kernelTask.put(kernelTask.kernelSize);
        kernelTask.put(kernelTask.input);
        kernelTask.put(kernelTask.output);
//        Thread.sleep(1);
        kernelTask.exe(device);
        join(kernelTask);
        kernelTask.get(kernelTask.weights);
        kernelTask.cleanUpArrays();
      } catch (Throwable e) {
        throw new RuntimeException("Error with " +this,e);
      }
    });
  }
  
  public void before() {
    //System.gc();
    //System.runFinalization();
  }
  
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append("Convolve [");
    builder.append(Arrays.toString(this.inputSize));
    builder.append(" x ");
    builder.append(Arrays.toString(this.kernelSize));
    builder.append(" => ");
    builder.append(Arrays.toString(this.outputSize));
    builder.append("]");
    return builder.toString();
  }
  
}
