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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.layers.opencl.BackpropKernel;
import com.simiacryptus.mindseye.layers.opencl.GradientKernel;
import com.simiacryptus.util.ml.Tensor;
import jcuda.jcudnn.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CONVOLUTION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

public final class ConvolutionController {
  
  private static final BackpropKernel backpropTask = new BackpropKernel();
  private static final GradientKernel kernelTask = new GradientKernel();
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ConvolutionController.class);
  
  private final int[] inputSize;
  private final int[] kernelSize;
  private final int[] outputSize;
  private final boolean simple;
  
  public int[] getOutputDims() {
    return outputSize;
  }
  
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
  
  public static int MAX_BUFFER_SIZE = 64 * 1024 * 1024;
  
  public void backprop(final double[][] input, final double[] weights, final double[][] output) {
    int length = input.length;
    assert(length == output.length);
    int inLength = input[0].length;
    int outLength = output[0].length;
    int inputsPerRun = Math.min(Math.floorDiv(MAX_BUFFER_SIZE, inLength), length);
    int runs = length / inputsPerRun;
    int leftover = length - runs * inputsPerRun;
    double[] inputBuffer = null;
    double[] outputBuffer = null;
    CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
            CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
    CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionDescriptor(
            simple?((kernelSize[1] - 1) / 2):0, simple?((kernelSize[0] - 1) / 2):0,
            1, 1,
            CUDNN_CONVOLUTION);
    CuDNN.CuDNNPtr filterData = CuDNN.write(weights);
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
      assert(0 < weights.length);
      assert(0 < outputBuffer.length);
      assert this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2] == weights.length;
      double[] _inputBuffer = inputBuffer;
      double[] _outputBuffer = outputBuffer;
      CuDNN.devicePool.with(device -> {
        try {
          CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
                  CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, currentNumItems, inputSize[2], inputSize[1], inputSize[0]);
          backprop(_inputBuffer, filterData, _outputBuffer, device, inputDescriptor, filterDescriptor, convolutionDescriptor);
        } catch (Throwable e) {
          throw new RuntimeException("Error with " + this,e);
        }
      });
      for (int i = 0; i< currentNumItems; i++) {
        assert inLength == input[currentIndexOffset+i].length;
        System.arraycopy(inputBuffer, i * inLength, input[currentIndexOffset+i], 0, inLength);
      }
    }
    filterData.finalize();
    Tensor.recycle(inputBuffer);
    Tensor.recycle(outputBuffer);
  }
  
  public void convolve(final double[][] input, final double[] weights, final double[][] output) {
    int length = input.length;
    assert(length == output.length);
    int inLength = input[0].length;
    int outLength = output[0].length;
    int inputsPerRun = Math.min(Math.floorDiv(MAX_BUFFER_SIZE, inLength), length);
    assert(0 < inputsPerRun) : "Requested buffer is over max of " + MAX_BUFFER_SIZE;
    int runs = length / inputsPerRun;
    int leftover = length - runs * inputsPerRun;
    double[] inputBuffer = null;
    double[] outputBuffer = null;
    CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
            CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
    CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionDescriptor(
            simple?((kernelSize[1] - 1) / 2):0, simple?((kernelSize[0] - 1) / 2):0,
            1, 1,
            CUDNN_CONVOLUTION);
    CuDNN.CuDNNPtr filterData = CuDNN.write(weights);
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
      assert(0 < weights.length);
      assert(0 < outputBuffer.length);
      double[] _inputBuffer = inputBuffer;
      double[] _outputBuffer = outputBuffer;
      CuDNN.devicePool.with(device -> {
        try {
          CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
                  CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, currentNumItems, inputSize[2], inputSize[1], inputSize[0]);
          convolve(_inputBuffer, filterData, _outputBuffer, device, inputDescriptor, filterDescriptor, convolutionDescriptor);
        } catch (Throwable e) {
          throw new RuntimeException("Error with " + this,e);
        }
      });
      for (int i = 0; i< currentNumItems; i++) {
        assert outLength == output[currentIndexOffset+i].length;
        System.arraycopy(outputBuffer, i * outLength, output[currentIndexOffset+i], 0, outLength);
      }
    }
    filterData.finalize();
    Tensor.recycle(inputBuffer);
    Tensor.recycle(outputBuffer);
  }
  
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
    CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
            CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
    CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionDescriptor(
            simple?((kernelSize[1] - 1) / 2):0, simple?((kernelSize[0] - 1) / 2):0,
            1, 1,
            CUDNN_CONVOLUTION);
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
      assert(0 < inputBuffer.length);
      assert(0 < buffer.length);
      assert(0 < outputBuffer.length);
      double[] _inputBuffer = inputBuffer;
      double[] _outputBuffer = outputBuffer;
      CuDNN.devicePool.with(device -> {
        try {
          int items = currentNumItems;
          CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
                  CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, items, inputSize[2], inputSize[1], inputSize[0]);
          gradient(_inputBuffer, buffer, _outputBuffer, device, inputDescriptor, filterDescriptor, convolutionDescriptor);
        } catch (Throwable e) {
          throw new RuntimeException("Error with " + this,e);
        }
      });
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

  private void backprop(double[] input, CuDNN.CuDNNPtr filterData, double[] output, CuDNN device, CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor, CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor, CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor) {
    int[] outputDims = device.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr());
    assert(4 == outputDims.length);
    assert(outputSize[0] == outputDims[3]);
    assert(outputSize[1] == outputDims[2]);
    assert(outputSize[2] == outputDims[1]);
    CuDNN.CuDNNResource<cudnnTensorDescriptor> outputDescriptor = device.newTensorDescriptor(
            CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, outputDims[0], outputDims[1], outputDims[2], outputDims[3]);
    int algorithm = device.getBackwardDataAlgorithm(
            inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
    CuDNN.CuDNNPtr workSpace = device.allocateBackwardDataWorkspace(
            inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
    CuDNN.CuDNNPtr alpha = device.javaPtr(1.0);
    CuDNN.CuDNNPtr beta = device.javaPtr(0.0);
    CuDNN.CuDNNPtr inputData = device.alloc(input);
    CuDNN.CuDNNPtr outputData = device.write(output);
    CuDNN.handle(cudnnConvolutionBackwardData(device.cudnnHandle, alpha.getPtr(),
            filterDescriptor.getPtr(), filterData.getPtr(),
            outputDescriptor.getPtr(), outputData.getPtr(),
            convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
            inputDescriptor.getPtr(), inputData.getPtr()));
    inputData.read(input);
    inputData.finalize();
    outputData.finalize();
  }

  private void convolve(double[] input, CuDNN.CuDNNPtr filterData, double[] output, CuDNN device, CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor, CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor, CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor) {
    int[] outputDims = device.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr());
    assert(4 == outputDims.length);
    assert(outputSize[0] == outputDims[3]);
    assert(outputSize[1] == outputDims[2]);
    assert(outputSize[2] == outputDims[1]);
    CuDNN.CuDNNResource<cudnnTensorDescriptor> outputDescriptor = device.newTensorDescriptor(
            CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, outputDims[0], outputDims[1], outputDims[2], outputDims[3]);
    int algorithm = device.getForwardAlgorithm(
            inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
    CuDNN.CuDNNPtr workSpace = device.allocateForwardWorkspace(
            inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
    CuDNN.CuDNNPtr alpha = device.javaPtr(1.0);
    CuDNN.CuDNNPtr beta = device.javaPtr(0.0);
    CuDNN.CuDNNPtr inputData = device.write(input);
    CuDNN.CuDNNPtr outputData = device.alloc(output);
    CuDNN.handle(cudnnConvolutionForward(device.cudnnHandle, alpha.getPtr(),
            inputDescriptor.getPtr(), inputData.getPtr(),
            filterDescriptor.getPtr(), filterData.getPtr(),
            convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
            outputDescriptor.getPtr(), outputData.getPtr()));
    outputData.read(output);
    inputData.finalize();
    outputData.finalize();
  }

  private void gradient(double[] input, double[] weights, double[] output, CuDNN device, CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor, CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor, CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor) {
    int[] outputDims = device.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr());
    assert(4 == outputDims.length);
    assert(outputSize[0] == outputDims[3]);
    assert(outputSize[1] == outputDims[2]);
    assert(outputSize[2] == outputDims[1]);
    CuDNN.CuDNNResource<cudnnTensorDescriptor> outputDescriptor = device.newTensorDescriptor(
            CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, outputDims[0], outputDims[1], outputDims[2], outputDims[3]);
    int algorithm = device.getBackwardFilterAlgorithm(
            inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
    CuDNN.CuDNNPtr workSpace = device.allocateBackwardFilterWorkspace(
            inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
    CuDNN.CuDNNPtr alpha = device.javaPtr(1.0);
    CuDNN.CuDNNPtr beta = device.javaPtr(0.0);
    CuDNN.CuDNNPtr inputData = device.write(input);
    CuDNN.CuDNNPtr filterData = device.alloc(weights);
    CuDNN.CuDNNPtr outputData = device.write(output);

    CuDNN.handle(cudnnConvolutionBackwardFilter(device.cudnnHandle, alpha.getPtr(),
            inputDescriptor.getPtr(), inputData.getPtr(),
            outputDescriptor.getPtr(), outputData.getPtr(),
            convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
            filterDescriptor.getPtr(), filterData.getPtr()));

    filterData.read(weights);
    inputData.finalize();
    filterData.finalize();
    outputData.finalize();
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
