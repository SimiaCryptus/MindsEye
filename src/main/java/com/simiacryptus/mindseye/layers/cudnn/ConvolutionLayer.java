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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardData;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardFilter;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CONVOLUTION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

public class ConvolutionLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("kernel", kernel.getJson());
    json.add("skip", skip.getJson());
    json.addProperty("simple", simple);
    return json;
  }
  
  public static ConvolutionLayer fromJson(JsonObject json) {
    return new ConvolutionLayer(json);
  }
  protected ConvolutionLayer(JsonObject json) {
    super(json);
    this.kernel = Tensor.fromJson(json.getAsJsonObject("kernel"));
    this.skip = Tensor.fromJson(json.getAsJsonObject("skip"));
    this.simple = json.getAsJsonPrimitive("simple").getAsBoolean();
  }
  
  
  public final Tensor kernel;
  public final Tensor skip;
  public final boolean simple;
  
  protected ConvolutionLayer() {
    this((Tensor)null, (Tensor)null, true);
  }
  
  protected ConvolutionLayer(Tensor kernel, Tensor skip, boolean simple) {
    super();
    this.simple = simple;
    this.skip = skip;
    if(kernel.getDimensions().length != 3) throw new IllegalArgumentException();
    if(kernel.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if(kernel.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if(kernel.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.kernel = kernel;
  }
  
  public ConvolutionLayer(final int width, int height, final int inputBands, final int outputBands) {
    this(width, height, inputBands * outputBands);
  }
  
  public ConvolutionLayer(final int width, int height, final int bands, boolean simple) {
    this(new Tensor(width,height,bands), new Tensor(new int[]{1,1}), simple);
    assert(!simple || 0 == (width-1) % 2) : "Simple kernels must have odd width";
    assert(!simple || 0 == (height-1) % 2) : "Simple kernels must have odd height";
  }
  
  public ConvolutionLayer(final int width, int height, final int bands) {
    this(width, height, bands, true);
  }
  
  public ConvolutionLayer(final int width, int height, final int inputBands, final int outputBands, boolean simple) {
    this(width, height, inputBands * outputBands, simple);
  }
  
  public ConvolutionLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.kernel.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    
    final NNResult input = inObj[0];
    final TensorList batch = input.data;
    final int[] inputSize = batch.get(0).getDimensions();
    int[] kernelSize = this.kernel.getDimensions();
    int[] outputSize = IntStream.range(0, kernelSize.length).map(i -> {
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
    Tensor[] output = IntStream.range(0, batch.length())
                           .mapToObj(dataIndex -> new Tensor(outputSize))
                           .toArray(i -> new Tensor[i]);
    try {
      double[][] inputBuffers = batch.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
      double[][] outputBuffers = Arrays.stream(output).map(x -> x.getData()).toArray(i -> new double[i][]);
      convolve(inputSize, kernelSize, outputSize, simple, inputBuffers, this.kernel.getData(), outputBuffers);
    } catch (Throwable e) {
      throw new RuntimeException("Error with image res " + Arrays.toString(inputSize),e);
    }
    assert Arrays.stream(output).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
  
    return new NNResult(output) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] error) {
        assert Arrays.stream(error).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        if (!isFrozen()) {
          double[][] inputBuffers = batch.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
          double[][] outputBuffers = Arrays.stream(error).map(x -> x.getData()).toArray(i -> new double[i][]);
          final Tensor kernel = ConvolutionLayer.this.kernel;
          final Tensor weightGradient = new Tensor(kernel.getDimensions());
          gradient(inputSize, kernelSize, outputSize, simple, inputBuffers, weightGradient.getData(), outputBuffers);
          buffer.get(ConvolutionLayer.this, kernel).accumulate(weightGradient.getData());
        }
        if (input.isAlive()) {
          Tensor[] inputBufferTensors = IntStream.range(0, data.length()).mapToObj(dataIndex -> new Tensor(inputSize)).toArray(i -> new Tensor[i]);
          double[][] inputBuffers = Arrays.stream(inputBufferTensors).map(x -> x.getData()).toArray(i -> new double[i][]);
          double[][] outputBuffers = Arrays.stream(error).map(x -> x.getData()).toArray(i -> new double[i][]);
          backprop(inputSize, kernelSize, outputSize, simple, inputBuffers, ConvolutionLayer.this.kernel.getData(), outputBuffers);
          assert Arrays.stream(inputBufferTensors).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          input.accumulate(buffer, inputBufferTensors);
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  public ConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.kernel.coordStream().parallel().forEach(c -> {
      this.kernel.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  public ConvolutionLayer setWeights(final DoubleSupplier f) {
    this.kernel.coordStream().parallel().forEach(c -> {
      this.kernel.set(c, f.getAsDouble());
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.kernel.getData());
  }

  public static int MAX_BUFFER_SIZE = 64 * 1024 * 1024;

  public static void backprop(final int[] inputSize, final int[] kernelSize, final int[] outputSize, boolean simple, final double[][] input, final double[] weights, final double[][] output) {
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
      assert kernelSize[0] * kernelSize[1] * kernelSize[2] == weights.length;
      double[] _inputBuffer = inputBuffer;
      double[] _outputBuffer = outputBuffer;
      CuDNN.devicePool.with(device -> {
        try {
          CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
                  CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, currentNumItems, inputSize[2], inputSize[1], inputSize[0]);
          backprop(outputSize, _inputBuffer, filterData, _outputBuffer, device, inputDescriptor, filterDescriptor, convolutionDescriptor);
        } catch (Throwable e) {
          throw new RuntimeException("Error with " + Arrays.toString(kernelSize),e);
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

  public static void convolve(final int[] inputSize, final int[] kernelSize, final int[] outputSize, boolean simple, final double[][] input, final double[] weights, final double[][] output) {
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
          convolve(outputSize, _inputBuffer, filterData, _outputBuffer, device, inputDescriptor, filterDescriptor, convolutionDescriptor);
        } catch (Throwable e) {
          throw new RuntimeException("Error with " + Arrays.toString(kernelSize),e);
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

  public static void gradient(final int[] inputSize, final int[] kernelSize, final int[] outputSize, boolean simple, final double[][] input, final double[] weights, final double[][] output) {
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
          gradient(outputSize, _inputBuffer, buffer, _outputBuffer, device, inputDescriptor, filterDescriptor, convolutionDescriptor);
        } catch (Throwable e) {
          throw new RuntimeException("Error with " + Arrays.toString(kernelSize),e);
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

  private static void backprop(final int[] outputSize, double[] input, CuDNN.CuDNNPtr filterData, double[] output, CuDNN device, CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor, CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor, CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor) {
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

  private static void convolve(final int[] outputSize, double[] input, CuDNN.CuDNNPtr filterData, double[] output, CuDNN device, CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor, CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor, CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor) {
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

  private static void gradient(final int[] outputSize, double[] input, double[] weights, double[] output, CuDNN device, CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor, CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor, CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor) {
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

}
