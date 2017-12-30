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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.Util;
import jcuda.jcudnn.*;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * This convolution layer only supports an equal number of input and output bands. It is used as the foundational
 * component for ConvolutionLayer, since the CuDNN api has this restriction (in recent versions).
 */
@SuppressWarnings("serial")
public class SimpleConvolutionLayer extends NNLayer implements LayerPrecision<SimpleConvolutionLayer> {
  
  
  /**
   * The Filter.
   */
  public final Tensor kernel;
  private int paddingX;
  private int paddingY;
  private Precision precision = Precision.Double;
  private int strideX = 1;
  private int strideY = 1;
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected SimpleConvolutionLayer() {
    this((Tensor) null);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   */
  public SimpleConvolutionLayer(final int width, final int height, final int bands) {
    this(new Tensor(width, height, bands));
    assert !false || 0 == (width - 1) % 2 : "Simple kernels must have odd width";
    assert !false || 0 == (height - 1) % 2 : "Simple kernels must have odd height";
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json the json
   */
  protected SimpleConvolutionLayer(final JsonObject json) {
    super(json);
    kernel = Tensor.fromJson(json.get("filter"));
    strideX = json.get("strideX").getAsInt();
    strideY = json.get("strideY").getAsInt();
    setPaddingX(json.get("paddingX").getAsInt());
    setPaddingY(json.get("paddingY").getAsInt());
    precision = Precision.valueOf(json.get("precision").getAsString());
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param kernel the filter
   */
  protected SimpleConvolutionLayer(final Tensor kernel) {
    super();
    int[] kernelSize = kernel.getDimensions();
    if (kernelSize.length != 3) throw new IllegalArgumentException();
    if (kernelSize[0] <= 0) throw new IllegalArgumentException();
    if (kernelSize[1] <= 0) throw new IllegalArgumentException();
    if (kernelSize[2] <= 0) throw new IllegalArgumentException();
    this.kernel = kernel;
    this.setPaddingX((int) Math.ceil((kernelSize[0] - 1) / 2.0));
    this.setPaddingY((int) Math.ceil((kernelSize[1] - 1) / 2.0));
  
  }
  
  /**
   * From json convolution layer.
   *
   * @param json the json
   * @return the convolution layer
   */
  public static SimpleConvolutionLayer fromJson(final JsonObject json) {
    return new SimpleConvolutionLayer(json);
  }
  
  /**
   * Reverse int [ ].
   *
   * @param array the array
   * @return the int [ ]
   */
  public static int[] reverse(int... array) {
    for (int i = 0; i < array.length / 2; i++) {
      int j = array[array.length - (i + 1)];
      array[array.length - (i + 1)] = array[i];
      array[i] = j;
    }
    return array;
  }
  
  /**
   * Add weights convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  public SimpleConvolutionLayer addWeights(final DoubleSupplier f) {
    Util.add(f, kernel.getData());
    return this;
  }
  
  private boolean cmp(final int[] outputSize, final int[] outputDims) {
    if (4 != outputDims.length) return false;
    if (outputSize[0] != outputDims[3]) return false;
    if (outputSize[1] != outputDims[2]) return false;
    return outputSize[2] == outputDims[1];
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    ((CudaExecutionContext) nncontext).initThread();
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.get(0).getDimensions();
    final int[] kernelSize = kernel.getDimensions();
    final int[] outputSize = getOutputSize(inputSize, kernelSize);
    final int length = batch.length();
    
    try {
  
      final CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      int outputChannels = kernelSize[2] / inputSize[2];
      final CudaResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, outputChannels, inputSize[2], kernelSize[1], kernelSize[0]);
      final CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionNdDescriptor(cudnnConvolutionMode.CUDNN_CONVOLUTION, precision.code,
                                                                                                              new int[]{getPaddingX(), getPaddingY()},
                                                                                                              new int[]{strideY, strideX},
                                                                                                              new int[]{1, 1});
      final CudaPtr alpha = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 1.0);
      final CudaPtr beta = precision.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 0.0);
  
      int[] outputDims = reverse(CuDNN.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr()));
      outputDims = IntStream.of(outputDims).limit(3).toArray();
      final CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, cudnnTensorFormat.CUDNN_TENSOR_NCHW, length, outputDims[2], outputDims[1], outputDims[0]);
  
      final double[] filterData = kernel.getData();
      final CudaPtr filterPtr = new CudaPtr(filterData.length * precision.size, ((CudaExecutionContext) nncontext).getDeviceNumber()).write(precision, filterData);
      assert 0 < filterData.length;
      final CudaPtr inputData = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, batch);
      assert kernelSize[0] * kernelSize[1] * kernelSize[2] == filterData.length;
  
      final CudaPtr outputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.dim(outputDims) * 1l * length * precision.size);
      final cudnnHandle cudnnHandle = ((CuDNN) nncontext).cudnnHandle;
      final int algorithm = ((CudaExecutionContext) nncontext).getForwardAlgorithm(
        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
      final CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateForwardWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                                                                                            inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
      CuDNN.handle(CuDNN.cudnnConvolutionForward(cudnnHandle, alpha.getPtr(),
                                                 inputDescriptor.getPtr(), inputData.getPtr(),
                                                 filterDescriptor.getPtr(), filterPtr.getPtr(),
                                                 convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                                                 outputDescriptor.getPtr(), outputBuffer.getPtr()));
      workSpace.finalize();
  
      TensorList output = new GpuTensorList(outputBuffer, length, outputDims, cudnnHandle, precision);
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
          ((CudaExecutionContext) nncontext).initThread();
          assert error.length() == batch.length();
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          final int length = error.length();
          final CudaPtr errorPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, error);
          if (!isFrozen()) {
            final CudaPtr filterBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), filterData.length * 1l * precision.size);
            try {
              final int backwardAlgorithm = ((CudaExecutionContext) nncontext).getBackwardFilterAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              final CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardFilterWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                                                                                                           inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), backwardAlgorithm);
              CuDNN.handle(CuDNN.cudnnConvolutionBackwardFilter(cudnnHandle, alpha.getPtr(),
                                                                inputDescriptor.getPtr(), inputData.getPtr(),
                                                                outputDescriptor.getPtr(), errorPtr.getPtr(),
                                                                convolutionDescriptor.getPtr(), backwardAlgorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                                                                filterDescriptor.getPtr(), filterBuffer.getPtr()));
              workSpace.finalize();
            } catch (final Throwable e) {
              throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputSize)), e);
            }
            final Tensor weightGradient = CudaPtr.read(filterBuffer, precision, kernel.getDimensions());
            buffer.get(SimpleConvolutionLayer.this, kernel.getData()).addInPlace(weightGradient.getData());
          }
          if (input.isAlive()) {
            final CudaPtr inputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), batch.get(0).dim() * 1l * length * precision.size);
            try {
              final int algorithm = ((CudaExecutionContext) nncontext).getBackwardDataAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              final CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardDataWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                                                                                                         inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
              CuDNN.handle(CuDNN.cudnnConvolutionBackwardData(cudnnHandle, alpha.getPtr(),
                                                              filterDescriptor.getPtr(), filterPtr.getPtr(),
                                                              outputDescriptor.getPtr(), errorPtr.getPtr(),
                                                              convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                                                              inputDescriptor.getPtr(), inputBuffer.getPtr()));
            } catch (final Throwable e) {
              throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputSize)), e);
            }
            final TensorList inputBufferTensors = new GpuTensorList(inputBuffer, length, inputSize, cudnnHandle, precision);
            input.accumulate(buffer, inputBufferTensors);
          }
        }
        
        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (final Throwable e) {
      throw new ComponentException(String.format("Error in convolution %s x %s", Arrays.toString(inputSize), Arrays.toString(kernelSize)), e);
    }
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    json.add("filter", kernel.toJson());
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("paddingX", getPaddingX());
    json.addProperty("paddingY", getPaddingY());
    json.addProperty("simple", false);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  
  /**
   * Get output size int [ ].
   *
   * @param inputSize  the input size
   * @param kernelSize the kernel size
   * @return the int [ ]
   */
  protected int[] getOutputSize(final int[] inputSize, final int[] kernelSize) {
    return IntStream.range(0, kernelSize.length).map(i -> {
      int x;
      if (i == kernelSize.length - 1) {
        //assert kernelSize[i] == inputSize[i];
        x = kernelSize[i] / inputSize[i];
      }
      else {
        x = inputSize[i];
      }
      assert 0 < x;
      return x;
    }).toArray();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public SimpleConvolutionLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  /**
   * The Stride x.
   *
   * @return the stride x
   */
  public int getStrideX() {
    return strideX;
  }
  
  /**
   * Sets stride x.
   *
   * @param strideX the stride x
   * @return the stride x
   */
  public SimpleConvolutionLayer setStrideX(final int strideX) {
    this.strideX = strideX;
    return this;
  }
  
  /**
   * The Stride y.
   *
   * @return the stride y
   */
  public int getStrideY() {
    return strideY;
  }
  
  /**
   * Sets stride y.
   *
   * @param strideY the stride y
   * @return the stride y
   */
  public SimpleConvolutionLayer setStrideY(final int strideY) {
    this.strideY = strideY;
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public SimpleConvolutionLayer setWeights(final DoubleSupplier f) {
    kernel.coordStream().parallel().forEach(c -> {
      kernel.set(c, f.getAsDouble());
    });
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public SimpleConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    kernel.coordStream().parallel().forEach(c -> {
      kernel.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(kernel.getData());
  }
  
  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  public int getPaddingX() {
    return paddingX;
  }
  
  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   * @return the padding x
   */
  public SimpleConvolutionLayer setPaddingX(int paddingX) {
    this.paddingX = paddingX;
    return this;
  }
  
  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  public int getPaddingY() {
    return paddingY;
  }
  
  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   * @return the padding y
   */
  public SimpleConvolutionLayer setPaddingY(int paddingY) {
    this.paddingY = paddingY;
    return this;
  }
}
