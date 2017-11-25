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

package com.simiacryptus.mindseye.layers.cudnn.f32;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CONVOLUTION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Convolution layer.
 */
public class SimpleConvolutionLayer extends NNLayer {
  
  /**
   * The Filter.
   */
  public final Tensor filter;
  private transient Map<Integer, GPUDataMirror> stateCache = new HashMap<>();
  private int strideX = 1;
  private int strideY = 1;
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json the json
   */
  protected SimpleConvolutionLayer(JsonObject json) {
    super(json);
    this.filter = Tensor.fromJson(json.getAsJsonObject("filter"));
    this.strideX = json.get("strideX").getAsInt();
    this.strideY = json.get("strideY").getAsInt();
  }
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected SimpleConvolutionLayer() {
    this((Tensor)null);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *  @param filter the filter
   *
   */
  protected SimpleConvolutionLayer(Tensor filter) {
    super();
    if (filter.getDimensions().length != 3) throw new IllegalArgumentException();
    if (filter.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if (filter.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if (filter.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.filter = filter;
  }
  
  /**
   * Instantiates a new Convolution layer.
   *  @param width  the width
   * @param height the height
   * @param bands  the bands
   */
  public SimpleConvolutionLayer(final int width, int height, final int bands) {
    this(new Tensor(width, height, bands));
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
  public SimpleConvolutionLayer(final int width, int height, final int inputBands, final int outputBands, boolean simple) {
    this(width, height, inputBands * outputBands);
  }
  
  /**
   * From json convolution layer.
   *
   * @param json the json
   * @return the convolution layer
   */
  public static SimpleConvolutionLayer fromJson(JsonObject json) {
    return new SimpleConvolutionLayer(json);
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("filter", filter.getJson());
    json.addProperty("simple", false);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    return json;
  }
  
  /**
   * Add weights convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  public SimpleConvolutionLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.filter.getData());
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    ((CudaExecutionContext) nncontext).initThread();
    final TensorList input = inObj[0].getData();
    final int[] inputSize = input.getDimensions();
    int[] kernelSize = this.filter.getDimensions();
    int length = input.length();
    
    try {
      
      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CudaResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernelSize[2] / inputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
      int paddingX = ((kernelSize[1] - 1) / 2);
      int paddingY = ((kernelSize[0] - 1) / 2);
//      CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionNdDescriptor(CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT,
//        new int[]{paddingX, paddingY},
//        new int[]{strideY, strideX},
//        new int[]{1, 1});
      CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutions2dDescriptor(paddingX, paddingY,strideY, strideX,CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
      int[] outputDims = CuDNN.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr());
      CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, outputDims[1], outputDims[2], outputDims[3]);
      CudaPtr alpha = CuDNN.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 1.0f);
      CudaPtr beta = CuDNN.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 0.0f);
      
      CudaPtr filterPtr = getStateCache().computeIfAbsent(((CudaExecutionContext) nncontext).getDeviceNumber(), i -> new GPUDataMirror(filter.dim()))
        .uploadAsFloats(((CudaExecutionContext) nncontext).getDeviceNumber(), filter.getData());
      CudaPtr inputData = inObj[0].getGpuFloats(((CudaExecutionContext) nncontext).getDeviceNumber());
      CudaPtr outputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.dim(outputDims) * 1l * Sizeof.FLOAT);
      try {
        int forwardAlgorithm = ((CudaExecutionContext) nncontext).getForwardAlgorithm(
          inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
        CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateForwardWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
          inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), forwardAlgorithm);
        CuDNN.handle(cudnnConvolutionForward(((CuDNN) nncontext).cudnnHandle, alpha.getPtr(),
          inputDescriptor.getPtr(), inputData.getPtr(),
          filterDescriptor.getPtr(), filterPtr.getPtr(),
          convolutionDescriptor.getPtr(), forwardAlgorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
          outputDescriptor.getPtr(), outputBuffer.getPtr()));
        workSpace.finalize();
      } catch (Throwable e) {
        throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputDims)), e);
      }
      TensorList output = CudaPtr.fromDeviceFloat(outputBuffer, length, new int[]{outputDims[3], outputDims[2], outputDims[1]}, ((CuDNN) nncontext).cudnnHandle);
      
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          outputBuffer.finalize();
          ((CudaExecutionContext) nncontext).initThread();
          assert (error.length() == input.length());
          int length = error.length();
          CudaPtr errorPtr = CudaPtr.toDeviceAsFloat(((CudaExecutionContext) nncontext).getDeviceNumber(), error);
          if (!isFrozen()) {
            CudaPtr filterBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), filter.dim() * 1l * Sizeof.FLOAT);
            try {
              int backwardAlgorithm = ((CudaExecutionContext) nncontext).getBackwardFilterAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardFilterWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), backwardAlgorithm);
              CuDNN.handle(cudnnConvolutionBackwardFilter(((CuDNN) nncontext).cudnnHandle, alpha.getPtr(),
                inputDescriptor.getPtr(), inputData.getPtr(),
                outputDescriptor.getPtr(), errorPtr.getPtr(),
                convolutionDescriptor.getPtr(), backwardAlgorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                filterDescriptor.getPtr(), filterBuffer.getPtr()));
              workSpace.finalize();
            } catch (Throwable e) {
              throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputDims)), e);
            }
            final Tensor weightGradient = CudaPtr.fromDeviceFloat(filterBuffer, SimpleConvolutionLayer.this.filter.getDimensions());
            buffer.get(SimpleConvolutionLayer.this, SimpleConvolutionLayer.this.filter).accumulate(weightGradient.getData());
          }
          if (inObj[0].isAlive()) {
            CudaPtr inputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.dim(input.getDimensions()) * 1l * length * Sizeof.FLOAT);
            try {
              int algorithm = ((CudaExecutionContext) nncontext).getBackwardDataAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardDataWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
              CuDNN.handle(cudnnConvolutionBackwardData(((CuDNN) nncontext).cudnnHandle, alpha.getPtr(),
                filterDescriptor.getPtr(), filterPtr.getPtr(),
                outputDescriptor.getPtr(), errorPtr.getPtr(),
                convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                inputDescriptor.getPtr(), inputBuffer.getPtr()));
              workSpace.finalize();
            } catch (Throwable e) {
              throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(new int[]{outputDims[3], outputDims[2], outputDims[1]})), e);
            }
            TensorList inputBufferTensors = CudaPtr.fromDeviceFloat(inputBuffer, length, inputSize, ((CuDNN) nncontext).cudnnHandle);
            inObj[0].accumulate(buffer, inputBufferTensors);
            inputBuffer.finalize();
          }
        }
        
        @Override
        public boolean isAlive() {
          return inObj[0].isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new ComponentException("Error with image res " + Arrays.toString(inputSize), e);
    }
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public SimpleConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.filter.coordStream().parallel().forEach(c -> {
      this.filter.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  /**
   * Sets weights log.
   *
   * @param value the value
   * @return the weights log
   */
  public SimpleConvolutionLayer setWeightsLog(final double value) {
    this.filter.coordStream().parallel().forEach(c -> {
      double random = FastRandom.random();
      assert Double.isFinite(random);
      double v = (random - 0.5) * Math.pow(10, value);
      assert Double.isFinite(v);
      this.filter.set(c, v);
    });
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public SimpleConvolutionLayer setWeights(final DoubleSupplier f) {
    this.filter.coordStream().parallel().forEach(c -> {
      this.filter.set(c, f.getAsDouble());
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.filter.getData());
  }
  
  /**
   * Gets stride x.
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
  public SimpleConvolutionLayer setStrideX(int strideX) {
    this.strideX = strideX;
    return this;
  }
  
  /**
   * Sets stride xy.
   *
   * @param strideX the stride x
   * @param strideY the stride y
   * @return the stride xy
   */
  public SimpleConvolutionLayer setStrideXY(int strideX, int strideY) {
    this.strideX = strideX;
    this.strideY = strideY;
    return this;
  }
  
  /**
   * Gets stride y.
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
  public SimpleConvolutionLayer setStrideY(int strideY) {
    this.strideY = strideY;
    return this;
  }
  
  /**
   * Gets state cache.
   *
   * @return the state cache
   */
  protected Map<Integer, GPUDataMirror> getStateCache() {
    if (stateCache == null) stateCache = new HashMap<>();
    return stateCache;
  }
}
