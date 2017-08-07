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

package com.simiacryptus.mindseye.layers.aparapi;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * The type Convolution controller.
 */
public final class ConvolutionController {
  
  private static final BackpropKernel backpropTask = new BackpropKernel();
  private static final ConvolveKernel convolveTask = new ConvolveKernel();
  private static final GradientKernel kernelTask = new GradientKernel();
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ConvolutionController.class);
  
  private final int[] inputSize;
  private final int[] kernelSize;
  private final int[] outputSize;
  private final boolean simple;
  
  /**
   * Get output dims int [ ].
   *
   * @return the int [ ]
   */
  public int[] getOutputDims() {
    return outputSize;
  }
  
  /**
   * Instantiates a new Convolution controller.
   *
   * @param inputSize  the input size
   * @param kernelSize the kernel size
   * @param simple     the simple
   */
  public ConvolutionController(final int[] inputSize, final int[] kernelSize, boolean simple) {
    this.inputSize = inputSize;
    this.kernelSize = kernelSize;
    this.simple = simple;
    this.outputSize = IntStream.range(0, kernelSize.length).map(i -> {
      int x;
      if (i == kernelSize.length - 1) {
        x = kernelSize[i] / inputSize[i];
      } else if(simple) {
        x = inputSize[i];
      } else {
        x = 1 + inputSize[i] - kernelSize[i];
      }
      if (0 >= x) {
        assert false;
      }
      return x;
    }).toArray();
    assert this.outputSize.length == 3;
    assert this.kernelSize.length == 3;
    assert this.inputSize.length == 3;
  }
  
  /**
   * The constant MAX_BUFFER_SIZE.
   */
  public static int MAX_BUFFER_SIZE = 1 * 1024 * 1024 / 2;
  
  /**
   * Backprop.
   *
   * @param input   the input
   * @param weights the weights
   * @param output  the output
   */
  public void backprop(final double[][] input, final double[] weights, final double[][] output) {
    int length = input.length;
    assert(length == output.length);
    int inLength = input[0].length;
    int outLength = output[0].length;
    int inputsPerRun = Math.min(Math.floorDiv(MAX_BUFFER_SIZE, inLength), length);
    int runs = length / inputsPerRun;
    int leftover = length - runs * inputsPerRun;
    OpenCL.devicePool.with(device -> {
      try {
        synchronized (backpropTask) {
          assert(0 < weights.length);
          assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == weights.length;
          backpropTask.setExplicit(true);
          backpropTask.weights = weights;
          backpropTask.put(backpropTask.weights);
          backpropTask.kernelSize = this.kernelSize;
          backpropTask.put(backpropTask.kernelSize);
          backpropTask.kernelOffset = new int[]{
              simple?((this.kernelSize[1] - 1) / 2):0,
              simple?((this.kernelSize[0] - 1) / 2):0
          };
          backpropTask.put(convolveTask.kernelOffset);
          double[] inputBuffer = null;
          double[] outputBuffer = null;
          for(int run=0;run<runs;run++) {
            int currentIndexOffset = run * inputsPerRun;
            int currentNumItems = run < run - 1 ? inputsPerRun : leftover == 0 ? inputsPerRun : leftover;
            if(null == inputBuffer || inputBuffer.length != inLength * currentNumItems) {
              Tensor.recycle(inputBuffer);
              inputBuffer = Tensor.obtain(inLength * currentNumItems);
            }
            if(null == outputBuffer || outputBuffer.length != outLength * currentNumItems) {
              Tensor.recycle(outputBuffer);
              outputBuffer = Tensor.obtain(outLength * currentNumItems);
            }
            for (int i = 0; i< currentNumItems; i++) {
              assert outLength == output[currentIndexOffset+i].length;
              System.arraycopy(output[currentIndexOffset+i], 0, outputBuffer, i * outLength, outLength);
            }
            assert(0 < inputBuffer.length);
            assert(0 < outputBuffer.length);
            backpropTask.input = inputBuffer;
            backpropTask.output = outputBuffer;
            backpropTask.outputSize = this.outputSize;
            backpropTask.inputSize = this.inputSize;
            backpropTask.put(backpropTask.outputSize);
            backpropTask.put(backpropTask.inputSize);
            backpropTask.put(backpropTask.output);
            backpropTask.exe(device);
            backpropTask.get(backpropTask.input);
            backpropTask.input = null;
            backpropTask.output = null;
            backpropTask.outputSize = null;
            backpropTask.inputSize = null;
            for (int i = 0; i< currentNumItems; i++) {
              assert inLength == input[currentIndexOffset+i].length;
              System.arraycopy(inputBuffer, i * inLength, input[currentIndexOffset+i], 0, inLength);
            }
          }
          Tensor.recycle(inputBuffer);
          Tensor.recycle(outputBuffer);
          backpropTask.kernelSize = null;
          backpropTask.weights = null;
        }
      } catch (Throwable e) {
        throw new RuntimeException("Error map " + this,e);
      }
    });

  }
  
  /**
   * Convolve.
   *
   * @param input   the input
   * @param weights the weights
   * @param output  the output
   */
  public void convolve(final double[][] input, final double[] weights, final double[][] output) {
    int length = input.length;
    assert(length == output.length);
    int inLength = input[0].length;
    int outLength = output[0].length;
    int inputsPerRun = Math.min(Math.floorDiv(MAX_BUFFER_SIZE, inLength), length);
    assert(0 < inputsPerRun) : "Requested buffer is over max of " + MAX_BUFFER_SIZE;
    int runs = length / inputsPerRun;
    int leftover = length - runs * inputsPerRun;
    OpenCL.devicePool.with(device -> {
      try {
        synchronized (convolveTask) {
          assert(null != weights);
          assert(0 < weights.length);
          convolveTask.setExplicit(true);
          convolveTask.weights = weights;
          convolveTask.put(convolveTask.weights);
          convolveTask.kernelSize = this.kernelSize;
          convolveTask.kernelOffset = new int[]{
              simple?((this.kernelSize[1] - 1) / 2):0,
              simple?((this.kernelSize[0] - 1) / 2):0
          };
          convolveTask.put(convolveTask.kernelOffset);
          convolveTask.put(convolveTask.kernelSize);
          double[] inputBuffer = null;
          double[] outputBuffer = null;
          for(int run=0;run<=runs;run++) {
            int currentIndexOffset = run * inputsPerRun;
            int currentNumItems = run < runs ? inputsPerRun : leftover;
            if(0 == currentNumItems) continue;
            if(null == inputBuffer || inputBuffer.length != inLength * currentNumItems) {
              Tensor.recycle(inputBuffer);
              inputBuffer = Tensor.obtain(inLength * currentNumItems);
            }
            if(null == outputBuffer || outputBuffer.length != outLength * currentNumItems) {
              Tensor.recycle(outputBuffer);
              outputBuffer = Tensor.obtain(outLength * currentNumItems);
            }
            for (int i = 0; i< currentNumItems; i++) {
              assert inLength == input[currentIndexOffset+i].length;
              System.arraycopy(input[currentIndexOffset+i], 0, inputBuffer, i * inLength, inLength);
            }
            assert(0 < inputBuffer.length);
            assert(0 < outputBuffer.length);
            convolveTask.input = inputBuffer;
            convolveTask.output = outputBuffer;
            convolveTask.outputSize = this.outputSize;
            convolveTask.inputSize = this.inputSize;
            convolveTask.put(convolveTask.outputSize);
            convolveTask.put(convolveTask.inputSize);
            convolveTask.put(convolveTask.input);
            convolveTask.exe(device);
            convolveTask.get(convolveTask.output);
            convolveTask.input = null;
            convolveTask.output = null;
            convolveTask.outputSize = null;
            convolveTask.inputSize = null;
            for (int i = 0; i< currentNumItems; i++) {
              assert outLength == output[currentIndexOffset+i].length;
              System.arraycopy(outputBuffer, i * outLength, output[currentIndexOffset+i], 0, outLength);
            }
          }
          Tensor.recycle(inputBuffer);
          Tensor.recycle(outputBuffer);
          convolveTask.kernelSize = null;
          convolveTask.weights = null;
        }
      } catch (Throwable e) {
        throw new RuntimeException("Error map " + this,e);
      }});
  }
  
  /**
   * Gradient.
   *
   * @param input   the input
   * @param weights the weights
   * @param output  the output
   */
  public void gradient(final double[][] input, final double[] weights, final double[][] output) {
    int length = input.length;
    assert(length == output.length);
    int inLength = input[0].length;
    int outLength = output[0].length;
    int inputsPerRun = Math.min(Math.floorDiv(MAX_BUFFER_SIZE, Math.max(inLength,outLength)), length);
    int runs = length / inputsPerRun;
    int leftover = length - runs * inputsPerRun;
    double[] inputBuffer = null;
    double[] outputBuffer = null;
    for(int run=0;run<runs;run++) {
      int currentIndexOffset = run * inputsPerRun;
      int currentNumItems = run < run - 1 ? inputsPerRun : leftover == 0 ? inputsPerRun : leftover;
      if(null == inputBuffer || inputBuffer.length != inLength * currentNumItems) {
        Tensor.recycle(inputBuffer);
        inputBuffer = Tensor.obtain(inLength * currentNumItems);
      }
      if(null == outputBuffer || outputBuffer.length != outLength * currentNumItems) {
        Tensor.recycle(outputBuffer);
        outputBuffer = Tensor.obtain(outLength * currentNumItems);
      }
      for (int i = 0; i< currentNumItems; i++) {
        assert inLength == input[currentIndexOffset+i].length;
        assert outLength == output[currentIndexOffset+i].length;
        System.arraycopy(input[currentIndexOffset+i], 0, inputBuffer, i * inLength, inLength);
        System.arraycopy(output[currentIndexOffset+i], 0, outputBuffer, i * outLength, outLength);
      }
      int parallelism = Math.min(16, inLength);
      double[] buffer = Tensor.obtain(weights.length * parallelism);
      gradient(inputBuffer,buffer, weights.length, outputBuffer);
      IntStream.range(0, weights.length).forEach(weightIndex -> {
        for (int i = weightIndex; i < buffer.length; i += weights.length) {
          weights[weightIndex] += buffer[i];
        }
      });
      Tensor.recycle(buffer);
    }
    Tensor.recycle(inputBuffer);
    Tensor.recycle(outputBuffer);
  }

  private void gradient(final double[] input, final double[] weights, int weightSize, final double[] output) {
    assert(0 < input.length);
    assert(0 < weights.length);
    assert(0 < output.length);
    OpenCL.devicePool.with(device -> {
      try {
        synchronized (kernelTask) {
          kernelTask.input = input;
          kernelTask.weights = weights;
          kernelTask.output = output;
          kernelTask.outputSize = this.outputSize;
          kernelTask.inputSize = this.inputSize;
          kernelTask.kernelSize = this.kernelSize;
          kernelTask.weightSize = weightSize;
          kernelTask.paralellism = weights.length / weightSize;
          kernelTask.kernelOffset = new int[]{
              simple?((this.kernelSize[1] - 1) / 2):0,
              simple?((this.kernelSize[0] - 1) / 2):0
          };
          kernelTask.setExplicit(true);
          kernelTask.put(convolveTask.kernelOffset);
          kernelTask.put(kernelTask.outputSize);
          kernelTask.put(kernelTask.inputSize);
          kernelTask.put(kernelTask.kernelSize);
          kernelTask.put(kernelTask.input);
          kernelTask.put(kernelTask.output);
          kernelTask.exe(device);
          kernelTask.get(kernelTask.weights);
          kernelTask.input = null;
          kernelTask.weights = null;
          kernelTask.output = null;
          kernelTask.outputSize = null;
          kernelTask.inputSize = null;
          kernelTask.kernelSize = null;
        }
      } catch (Throwable e) {
        throw new RuntimeException("Error map " +this,e);
      }
    });
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
