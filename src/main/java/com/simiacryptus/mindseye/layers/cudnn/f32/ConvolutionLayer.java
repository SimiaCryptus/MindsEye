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
import com.simiacryptus.mindseye.data.Coordinate;
import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.data.TensorList;
import com.simiacryptus.mindseye.lang.ComponentException;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.CudaResource;
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
public class ConvolutionLayer extends NNLayer {

  private transient Map<Integer, GPUDataMirror> stateCache = new HashMap<>();
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("filter", filter.getJson());
    json.addProperty("simple", simple);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    return json;
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
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json the json
   */
  protected ConvolutionLayer(JsonObject json) {
    super(json);
    this.filter = Tensor.fromJson(json.getAsJsonObject("filter"));
    this.simple = json.get("simple").getAsBoolean();
    this.strideX = json.get("strideX").getAsInt();
    this.strideY = json.get("strideY").getAsInt();
  }
  
  
  /**
   * The Filter.
   */
  public final Tensor filter;
  /**
   * The Simple.
   */
  public final boolean simple;
  private int strideX = 1;
  private int strideY = 1;
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected ConvolutionLayer() {
    this((Tensor) null, (Tensor) null, true);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param filter the filter
   * @param skip   the skip
   * @param simple the simple
   */
  protected ConvolutionLayer(Tensor filter, Tensor skip, boolean simple) {
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
    this(new Tensor(width, height, bands), new Tensor(new int[]{1, 1}), simple);
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
      CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionDescriptor(
        simple ? ((kernelSize[1] - 1) / 2) : 0, simple ? ((kernelSize[0] - 1) / 2) : 0,
        strideX, strideY, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
      int[] outputDims = CuDNN.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr());
      int[] outputSize = new int[]{outputDims[3], outputDims[2], outputDims[1]};
      CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, outputSize[2], outputSize[1], outputSize[0]);
      CudaPtr alpha = CuDNN.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 1.0f);
      CudaPtr beta = CuDNN.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), 0.0f);

      CudaPtr filterPtr = getStateCache().computeIfAbsent(((CudaExecutionContext) nncontext).getDeviceNumber(), i -> new GPUDataMirror(filter.dim()))
                            .uploadAsFloats(((CudaExecutionContext) nncontext).getDeviceNumber(), filter.getData());
      CudaPtr inputData = inObj[0].getGpuFloats(((CudaExecutionContext) nncontext).getDeviceNumber());
      CudaPtr outputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.dim(outputSize) * 1l * length * Sizeof.FLOAT);
      try {
        int algorithm = ((CudaExecutionContext) nncontext).getForwardAlgorithm(
          inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
        CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateForwardWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
          inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
        CuDNN.handle(cudnnConvolutionForward(((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle, alpha.getPtr(),
          inputDescriptor.getPtr(), inputData.getPtr(),
          filterDescriptor.getPtr(), filterPtr.getPtr(),
          convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
          outputDescriptor.getPtr(), outputBuffer.getPtr()));
        workSpace.finalize();
      } catch (Throwable e) {
        throw new ComponentException("Error with " + Arrays.toString(kernelSize), e);
      }
      TensorList output = CudaPtr.fromDeviceFloat(outputBuffer, length, outputSize, ((CuDNN) nncontext).cudnnHandle);

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
              int algorithm = ((CudaExecutionContext) nncontext).getBackwardFilterAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardFilterWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
              CuDNN.handle(cudnnConvolutionBackwardFilter(((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle, alpha.getPtr(),
                inputDescriptor.getPtr(), inputData.getPtr(),
                outputDescriptor.getPtr(), errorPtr.getPtr(),
                convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                filterDescriptor.getPtr(), filterBuffer.getPtr()));
              workSpace.finalize();
            } catch (Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(kernelSize), e);
            }
            buffer.get(ConvolutionLayer.this, () -> new CudnnFloatDelta(filter.getData(), ConvolutionLayer.this))
              .accumulate(CuDNN.newTensorDescriptor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, kernelSize[2], kernelSize[1], kernelSize[0]), filterBuffer, (CudaExecutionContext) nncontext);
          }
          if (inObj[0].isAlive()) {
            CudaPtr inputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.dim(input.getDimensions()) * 1l * length * Sizeof.FLOAT);
            try {
              int algorithm = ((CudaExecutionContext) nncontext).getBackwardDataAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardDataWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
              CuDNN.handle(cudnnConvolutionBackwardData(((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle, alpha.getPtr(),
                filterDescriptor.getPtr(), filterPtr.getPtr(),
                outputDescriptor.getPtr(), errorPtr.getPtr(),
                convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                inputDescriptor.getPtr(), inputBuffer.getPtr()));
              workSpace.finalize();
            } catch (Throwable e) {
              throw new ComponentException("Error with " + Arrays.toString(kernelSize), e);
            }
            TensorList inputBufferTensors = CudaPtr.fromDeviceFloat(inputBuffer, length, inputSize, ((CuDNN) ((CudaExecutionContext) nncontext)).cudnnHandle);
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
  public ConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
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
  public ConvolutionLayer setWeightsLog(final double value) {
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
  public ConvolutionLayer setStrideX(int strideX) {
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
  public ConvolutionLayer setStrideXY(int strideX, int strideY) {
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
  public ConvolutionLayer setStrideY(int strideY) {
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
