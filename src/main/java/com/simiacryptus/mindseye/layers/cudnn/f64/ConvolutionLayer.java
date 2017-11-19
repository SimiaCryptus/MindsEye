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

package com.simiacryptus.mindseye.layers.cudnn.f64;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.CudaResource;
import com.simiacryptus.util.Util;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CONVOLUTION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Convolution layer.
 */
public class ConvolutionLayer extends NNLayer {
  
  
  /**
   * The Filter.
   */
  public final Tensor filter;
  /**
   * The Simple.
   */
  public final boolean simple;
  /**
   * The Stride x.
   */
  int strideX = 1;
  /**
   * The Stride y.
   */
  int strideY = 1;
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json the json
   */
  protected ConvolutionLayer(JsonObject json) {
    super(json);
    this.filter = Tensor.fromJson(json.getAsJsonObject("filter"));
    this.strideX = json.get("strideX").getAsInt();
    this.strideY = json.get("strideY").getAsInt();
    this.simple = json.getAsJsonPrimitive("simple").getAsBoolean();
  }
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected ConvolutionLayer() {
    this(null, true);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param filter the filter
   * @param simple the simple
   */
  protected ConvolutionLayer(Tensor filter, boolean simple) {
    super();
    this.simple = simple;
    if (filter.getDimensions().length != 3) throw new IllegalArgumentException();
    if (filter.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if (filter.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if (filter.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.filter = filter;
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width       the width
   * @param height      the height
   * @param inputBands  the input bands
   * @param outputBands the output bands
   */
  public ConvolutionLayer(final int width, int height, final int inputBands, final int outputBands) {
    this(width, height, inputBands * outputBands);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   * @param simple the simple
   */
  public ConvolutionLayer(final int width, int height, final int bands, boolean simple) {
    this(new Tensor(width, height, bands), simple);
    assert (!simple || 0 == (width - 1) % 2) : "Simple kernels must have odd width";
    assert (!simple || 0 == (height - 1) % 2) : "Simple kernels must have odd height";
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   */
  public ConvolutionLayer(final int width, int height, final int bands) {
    this(width, height, bands, true);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width       the width
   * @param height      the height
   * @param inputBands  the input bands
   * @param outputBands the output bands
   * @param simple      the simple
   */
  public ConvolutionLayer(final int width, int height, final int inputBands, final int outputBands, boolean simple) {
    this(width, height, inputBands * outputBands, simple);
  }
  
  /**
   * From json convolution layer.
   *
   * @param json the json
   * @return the convolution layer
   */
  public static ConvolutionLayer fromJson(JsonObject json) {
    return new ConvolutionLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("filter", filter.getJson());
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("simple", simple);
    return json;
  }
  
  /**
   * Add weights convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  public ConvolutionLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.filter.getData());
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    ((CudaExecutionContext) nncontext).initThread();
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.get(0).getDimensions();
    int[] kernelSize = this.filter.getDimensions();
    int[] outputSize = getOutputSize(inputSize, kernelSize);
    int length = batch.length();
    
    try {
      
      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CudaResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
        CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
      CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, outputSize[2], outputSize[1], outputSize[0]);
      CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionDescriptor(
        simple ? ((kernelSize[1] - 1) / 2) : 0, simple ? ((kernelSize[0] - 1) / 2) : 0,
        strideX, strideY,
        CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);
      CudaPtr alpha = CuDNN.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 1.0);
      CudaPtr beta = CuDNN.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 0.0);
      
      final double[] filterData = this.filter.getData();
      CudaPtr filterPtr = CuDNN.write(((CudaExecutionContext) nncontext).getDeviceNumber(), filterData);
      assert (0 < filterData.length);
      CudaPtr inputData = CudaPtr.toDeviceAsDouble(((CudaExecutionContext) nncontext).getDeviceNumber(), batch);
      assert kernelSize[0] * kernelSize[1] * kernelSize[2] == filterData.length;
      
      CudaPtr outputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.dim(outputSize) * 1l * length * Sizeof.DOUBLE);
      final cudnnHandle cudnnHandle = ((CuDNN) nncontext).cudnnHandle;
      try {
        assert verifyOutputDims(inputDescriptor, filterDescriptor, convolutionDescriptor, outputSize);
        int algorithm = ((CudaExecutionContext) nncontext).getForwardAlgorithm(
          inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
        CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateForwardWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
          inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
        CuDNN.handle(cudnnConvolutionForward(cudnnHandle, alpha.getPtr(),
          inputDescriptor.getPtr(), inputData.getPtr(),
          filterDescriptor.getPtr(), filterPtr.getPtr(),
          convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
          outputDescriptor.getPtr(), outputBuffer.getPtr()));
        workSpace.finalize();
      } catch (Throwable e) {
        throw new ComponentException("Error with " + Arrays.toString(kernelSize), e);
      }
      TensorList output = CudaPtr.fromDeviceDouble(outputBuffer, length, outputSize, cudnnHandle);
      
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          ((CudaExecutionContext) nncontext).initThread();
          assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          int length = error.length();
          CudaPtr errorPtr = CudaPtr.toDeviceAsDouble(((CudaExecutionContext) nncontext).getDeviceNumber(), error);
          if (!isFrozen()) {
            CudaPtr filterBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), filterData.length * 1l * Sizeof.DOUBLE);
            try {
              int algorithm = ((CudaExecutionContext) nncontext).getBackwardFilterAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardFilterWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
              CuDNN.handle(cudnnConvolutionBackwardFilter(cudnnHandle, alpha.getPtr(),
                inputDescriptor.getPtr(), inputData.getPtr(),
                outputDescriptor.getPtr(), errorPtr.getPtr(),
                convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                filterDescriptor.getPtr(), filterBuffer.getPtr()));
              workSpace.finalize();
            } catch (Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(kernelSize), e);
            }
            final Tensor weightGradient = CudaPtr.fromDeviceDouble(filterBuffer, ConvolutionLayer.this.filter.getDimensions());
            buffer.get(ConvolutionLayer.this, ConvolutionLayer.this.filter).accumulate(weightGradient.getData());
          }
          if (input.isAlive()) {
            CudaPtr inputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), batch.get(0).dim() * 1l * length * Sizeof.DOUBLE);
            try {
              int algorithm = ((CudaExecutionContext) nncontext).getBackwardDataAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardDataWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
              CuDNN.handle(cudnnConvolutionBackwardData(cudnnHandle, alpha.getPtr(),
                filterDescriptor.getPtr(), filterPtr.getPtr(),
                outputDescriptor.getPtr(), errorPtr.getPtr(),
                convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                inputDescriptor.getPtr(), inputBuffer.getPtr()));
            } catch (Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(kernelSize), e);
            }
            TensorList inputBufferTensors = CudaPtr.fromDeviceDouble(inputBuffer, length, inputSize, cudnnHandle);
            input.accumulate(buffer, inputBufferTensors);
          }
        }
        
        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
    }
  }
  
  /**
   * Get output size int [ ].
   *
   * @param inputSize  the input size
   * @param kernelSize the kernel size
   * @return the int [ ]
   */
  protected int[] getOutputSize(int[] inputSize, int[] kernelSize) {
    return IntStream.range(0, kernelSize.length).map(i -> {
      int x;
      if (i == kernelSize.length - 1) {
        x = kernelSize[i] / inputSize[i];
      }
      else if (simple) {
        x = inputSize[i];
      }
      else {
        x = 1 + inputSize[i] - kernelSize[i];
      }
      assert 0 < x;
      return x;
    }).toArray();
  }
  
  
  /**
   * Verify output dims boolean.
   *
   * @param inputDescriptor       the input descriptor
   * @param filterDescriptor      the filter descriptor
   * @param convolutionDescriptor the convolution descriptor
   * @param outputSize            the output size
   * @return the boolean
   */
  protected boolean verifyOutputDims(CudaResource<cudnnTensorDescriptor> inputDescriptor, CudaResource<cudnnFilterDescriptor> filterDescriptor, CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor, int[] outputSize) {
    int[] outputDims = CuDNN.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr());
    if (4 != outputDims.length) return false;
    if (outputSize[0] != outputDims[3]) return false;
    if (outputSize[1] != outputDims[2]) return false;
    return outputSize[2] == outputDims[1];
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public ConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.filter.coordStream().parallel().forEach(c -> {
      this.filter.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public ConvolutionLayer setWeights(final DoubleSupplier f) {
    this.filter.coordStream().parallel().forEach(c -> {
      this.filter.set(c, f.getAsDouble());
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.filter.getData());
  }
  
}
