/*
 * Copyright (c) 2018 by Andrew Charneski.
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

import com.simiacryptus.mindseye.lang.ComponentException;
import com.simiacryptus.mindseye.lang.RecycleBin;
import org.jetbrains.annotations.Nullable;
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
  /**
   * The constant MAX_BUFFER_SIZE.
   */
  public static int MAX_BUFFER_SIZE = 1 * 1024 * 1024 / 2;
  private final int[] inputSize;
  @javax.annotation.Nonnull
  private final int[] kernelSize;
  private final int[] outputSize;
  private @Nullable Integer paddingX = null;
  private @Nullable Integer paddingY = null;
  
  /**
   * Instantiates a new Convolution controller.
   *
   * @param inputSize  the input size
   * @param kernelSize the kernel size
   * @param paddingX   the padding x
   * @param paddingY   the padding y
   */
  public ConvolutionController(final int[] inputSize, @javax.annotation.Nonnull final int[] kernelSize, final Integer paddingX, Integer paddingY) {
    this.inputSize = inputSize;
    this.kernelSize = kernelSize;
    this.setPaddingX(paddingX);
    this.setPaddingY(paddingY);
    outputSize = IntStream.range(0, kernelSize.length).map(i -> {
      int x;
      @Nullable Integer padding;
      if (i == 0) {
        padding = paddingX;
      }
      else if (i == 1) {
        padding = paddingY;
      }
      else {
        padding = null;
      }
      if (i == kernelSize.length - 1) {
        x = kernelSize[i] / inputSize[i];
      }
      else if (null == padding) {
        x = inputSize[i];
      }
      else {
        x = 1 + inputSize[i] - kernelSize[i] + padding;
      }
      assert 0 < x;
      return x;
    }).toArray();
    assert outputSize.length == 3;
    assert this.kernelSize.length == 3;
    assert this.inputSize.length == 3;
  }
  
  /**
   * Backprop.
   *
   * @param input   the input
   * @param weights the weights
   * @param output  the output
   */
  public void backprop(@javax.annotation.Nonnull final double[][] input, @javax.annotation.Nonnull final double[] weights, @javax.annotation.Nonnull final double[][] output) {
    final int length = input.length;
    assert length == output.length;
    final int inLength = input[0].length;
    final int outLength = output[0].length;
    final int inputsPerRun = Math.min(Math.floorDiv(ConvolutionController.MAX_BUFFER_SIZE, inLength), length);
    final int runs = length / inputsPerRun;
    final int leftover = length - runs * inputsPerRun;
    OpenCL.devicePool.with(device -> {
      try {
        synchronized (ConvolutionController.backpropTask) {
          assert 0 < weights.length;
          assert kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length;
          ConvolutionController.backpropTask.setExplicit(true);
          ConvolutionController.backpropTask.weights = weights;
          ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.weights);
          ConvolutionController.backpropTask.kernelSize = kernelSize;
          ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.kernelSize);
          ConvolutionController.backpropTask.kernelOffset = new int[]{
            null == paddingY ? (kernelSize[1] - 1) / 2 : paddingY,
            null == paddingX ? (kernelSize[0] - 1) / 2 : paddingX
          };
          ConvolutionController.backpropTask.put(ConvolutionController.convolveTask.kernelOffset);
          @Nullable double[] inputBuffer = null;
          @Nullable double[] outputBuffer = null;
          for (int run = 0; run < runs; run++) {
            final int currentIndexOffset = run * inputsPerRun;
            final int currentNumItems = run < run - 1 ? inputsPerRun : leftover == 0 ? inputsPerRun : leftover;
            if (null == inputBuffer || inputBuffer.length != inLength * currentNumItems) {
              if (null != inputBuffer) RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
              inputBuffer = RecycleBin.DOUBLES.obtain(inLength * currentNumItems);
            }
            if (null == outputBuffer || outputBuffer.length != outLength * currentNumItems) {
              if (null != outputBuffer) RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
              outputBuffer = RecycleBin.DOUBLES.obtain(outLength * currentNumItems);
            }
            for (int i = 0; i < currentNumItems; i++) {
              assert outLength == output[currentIndexOffset + i].length;
              System.arraycopy(output[currentIndexOffset + i], 0, outputBuffer, i * outLength, outLength);
            }
            assert 0 < inputBuffer.length;
            assert 0 < outputBuffer.length;
            ConvolutionController.backpropTask.input = inputBuffer;
            ConvolutionController.backpropTask.output = outputBuffer;
            ConvolutionController.backpropTask.outputSize = outputSize;
            ConvolutionController.backpropTask.inputSize = inputSize;
            ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.outputSize);
            ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.inputSize);
            ConvolutionController.backpropTask.put(ConvolutionController.backpropTask.output);
            ConvolutionController.backpropTask.exe(device);
            ConvolutionController.backpropTask.get(ConvolutionController.backpropTask.input);
            ConvolutionController.backpropTask.input = null;
            ConvolutionController.backpropTask.output = null;
            ConvolutionController.backpropTask.outputSize = null;
            ConvolutionController.backpropTask.inputSize = null;
            for (int i = 0; i < currentNumItems; i++) {
              assert inLength == input[currentIndexOffset + i].length;
              System.arraycopy(inputBuffer, i * inLength, input[currentIndexOffset + i], 0, inLength);
            }
          }
          RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
          RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
          ConvolutionController.backpropTask.kernelSize = null;
          ConvolutionController.backpropTask.weights = null;
        }
      } catch (@javax.annotation.Nonnull final Throwable e) {
        throw new ComponentException("Error with " + this, e);
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
  public void convolve(@javax.annotation.Nonnull final double[][] input, @javax.annotation.Nonnull final double[] weights, @javax.annotation.Nonnull final double[][] output) {
    final int length = input.length;
    assert length == output.length;
    final int inLength = input[0].length;
    final int outLength = output[0].length;
    final int inputsPerRun = Math.min(Math.floorDiv(ConvolutionController.MAX_BUFFER_SIZE, inLength), length);
    assert 0 < inputsPerRun : "Requested buffer is over max of " + ConvolutionController.MAX_BUFFER_SIZE;
    final int runs = length / inputsPerRun;
    final int leftover = length - runs * inputsPerRun;
    OpenCL.devicePool.with(device -> {
      try {
        synchronized (ConvolutionController.convolveTask) {
          assert null != weights;
          assert 0 < weights.length;
          ConvolutionController.convolveTask.setExplicit(true);
          ConvolutionController.convolveTask.weights = weights;
          ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.weights);
          ConvolutionController.convolveTask.kernelSize = kernelSize;
          ConvolutionController.convolveTask.kernelOffset = new int[]{
            null == paddingY ? (kernelSize[1] - 1) / 2 : paddingY,
            null == paddingX ? (kernelSize[0] - 1) / 2 : paddingX
          };
          ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.kernelOffset);
          ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.kernelSize);
          @Nullable double[] inputBuffer = null;
          @Nullable double[] outputBuffer = null;
          for (int run = 0; run <= runs; run++) {
            final int currentIndexOffset = run * inputsPerRun;
            final int currentNumItems = run < runs ? inputsPerRun : leftover;
            if (0 == currentNumItems) {
              continue;
            }
            if (null == inputBuffer || inputBuffer.length != inLength * currentNumItems) {
              if (null != inputBuffer) RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
              inputBuffer = RecycleBin.DOUBLES.obtain(inLength * currentNumItems);
            }
            if (null == outputBuffer || outputBuffer.length != outLength * currentNumItems) {
              if (null != outputBuffer) RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
              outputBuffer = RecycleBin.DOUBLES.obtain(outLength * currentNumItems);
            }
            for (int i = 0; i < currentNumItems; i++) {
              assert inLength == input[currentIndexOffset + i].length;
              System.arraycopy(input[currentIndexOffset + i], 0, inputBuffer, i * inLength, inLength);
            }
            assert 0 < inputBuffer.length;
            assert 0 < outputBuffer.length;
            ConvolutionController.convolveTask.input = inputBuffer;
            ConvolutionController.convolveTask.output = outputBuffer;
            ConvolutionController.convolveTask.outputSize = outputSize;
            ConvolutionController.convolveTask.inputSize = inputSize;
            ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.outputSize);
            ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.inputSize);
            ConvolutionController.convolveTask.put(ConvolutionController.convolveTask.input);
            ConvolutionController.convolveTask.exe(device);
            ConvolutionController.convolveTask.get(ConvolutionController.convolveTask.output);
            ConvolutionController.convolveTask.input = null;
            ConvolutionController.convolveTask.output = null;
            ConvolutionController.convolveTask.outputSize = null;
            ConvolutionController.convolveTask.inputSize = null;
            for (int i = 0; i < currentNumItems; i++) {
              assert outLength == output[currentIndexOffset + i].length;
              System.arraycopy(outputBuffer, i * outLength, output[currentIndexOffset + i], 0, outLength);
            }
          }
          RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
          RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
          ConvolutionController.convolveTask.kernelSize = null;
          ConvolutionController.convolveTask.weights = null;
        }
      } catch (@javax.annotation.Nonnull final Throwable e) {
        throw new ComponentException("Error with " + this, e);
      }
    });
  }
  
  /**
   * Get output dims int [ ].
   *
   * @return the int [ ]
   */
  public int[] getOutputDims() {
    return outputSize;
  }
  
  private void gradient(@javax.annotation.Nonnull final double[] input, @javax.annotation.Nonnull final double[] weights, final int weightSize, @javax.annotation.Nonnull final double[] output) {
    assert 0 < input.length;
    assert 0 < weights.length;
    assert 0 < output.length;
    OpenCL.devicePool.with(device -> {
      try {
        synchronized (ConvolutionController.kernelTask) {
          ConvolutionController.kernelTask.input = input;
          ConvolutionController.kernelTask.weights = weights;
          ConvolutionController.kernelTask.output = output;
          ConvolutionController.kernelTask.outputSize = outputSize;
          ConvolutionController.kernelTask.inputSize = inputSize;
          ConvolutionController.kernelTask.kernelSize = kernelSize;
          ConvolutionController.kernelTask.weightSize = weightSize;
          ConvolutionController.kernelTask.paralellism = weights.length / weightSize;
          ConvolutionController.kernelTask.kernelOffset = new int[]{
            paddingY == null ? (kernelSize[1] - 1) / 2 : paddingY,
            paddingX == null ? (kernelSize[0] - 1) / 2 : paddingX
          };
          ConvolutionController.kernelTask.setExplicit(true);
          ConvolutionController.kernelTask.put(ConvolutionController.convolveTask.kernelOffset);
          ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.outputSize);
          ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.inputSize);
          ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.kernelSize);
          ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.input);
          ConvolutionController.kernelTask.put(ConvolutionController.kernelTask.output);
          ConvolutionController.kernelTask.exe(device);
          ConvolutionController.kernelTask.get(ConvolutionController.kernelTask.weights);
          ConvolutionController.kernelTask.input = null;
          ConvolutionController.kernelTask.weights = null;
          ConvolutionController.kernelTask.output = null;
          ConvolutionController.kernelTask.outputSize = null;
          ConvolutionController.kernelTask.inputSize = null;
          ConvolutionController.kernelTask.kernelSize = null;
        }
      } catch (@javax.annotation.Nonnull final Throwable e) {
        throw new ComponentException("Error with " + this, e);
      }
    });
  }
  
  /**
   * Gradient.
   *
   * @param input   the input
   * @param weights the weights
   * @param output  the output
   */
  public void gradient(@javax.annotation.Nonnull final double[][] input, @javax.annotation.Nonnull final double[] weights, @javax.annotation.Nonnull final double[][] output) {
    final int length = input.length;
    assert length == output.length;
    final int inLength = input[0].length;
    final int outLength = output[0].length;
    final int inputsPerRun = Math.min(Math.floorDiv(ConvolutionController.MAX_BUFFER_SIZE, Math.max(inLength, outLength)), length);
    final int runs = length / inputsPerRun;
    final int leftover = length - runs * inputsPerRun;
    @Nullable double[] inputBuffer = null;
    @Nullable double[] outputBuffer = null;
    for (int run = 0; run < runs; run++) {
      final int currentIndexOffset = run * inputsPerRun;
      final int currentNumItems = run < run - 1 ? inputsPerRun : leftover == 0 ? inputsPerRun : leftover;
      if (null == inputBuffer || inputBuffer.length != inLength * currentNumItems) {
        if (null != inputBuffer) RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
        inputBuffer = RecycleBin.DOUBLES.obtain(inLength * currentNumItems);
      }
      if (null == outputBuffer || outputBuffer.length != outLength * currentNumItems) {
        if (null != outputBuffer) RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
        outputBuffer = RecycleBin.DOUBLES.obtain(outLength * currentNumItems);
      }
      for (int i = 0; i < currentNumItems; i++) {
        assert inLength == input[currentIndexOffset + i].length;
        assert outLength == output[currentIndexOffset + i].length;
        System.arraycopy(input[currentIndexOffset + i], 0, inputBuffer, i * inLength, inLength);
        System.arraycopy(output[currentIndexOffset + i], 0, outputBuffer, i * outLength, outLength);
      }
      final int parallelism = Math.min(16, inLength);
      final double[] buffer = RecycleBin.DOUBLES.obtain(weights.length * parallelism);
      gradient(inputBuffer, buffer, weights.length, outputBuffer);
      IntStream.range(0, weights.length).forEach(weightIndex -> {
        for (int i = weightIndex; i < buffer.length; i += weights.length) {
          weights[weightIndex] += buffer[i];
        }
      });
      RecycleBin.DOUBLES.recycle(buffer, buffer.length);
    }
    RecycleBin.DOUBLES.recycle(inputBuffer, inputBuffer.length);
    RecycleBin.DOUBLES.recycle(outputBuffer, outputBuffer.length);
  }
  
  @Override
  public String toString() {
    @javax.annotation.Nonnull final StringBuilder builder = new StringBuilder();
    builder.append("Convolve [");
    builder.append(Arrays.toString(inputSize));
    builder.append(" x ");
    builder.append(Arrays.toString(kernelSize));
    builder.append(" => ");
    builder.append(Arrays.toString(outputSize));
    builder.append("]");
    return builder.toString();
  }
  
  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  public @Nullable Integer getPaddingX() {
    return paddingX;
  }
  
  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   */
  public void setPaddingX(Integer paddingX) {
    this.paddingX = paddingX;
  }
  
  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  public @Nullable Integer getPaddingY() {
    return paddingY;
  }
  
  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   */
  public void setPaddingY(Integer paddingY) {
    this.paddingY = paddingY;
  }
}
