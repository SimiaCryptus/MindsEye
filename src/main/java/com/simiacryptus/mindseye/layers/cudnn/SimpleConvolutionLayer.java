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
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import static com.simiacryptus.mindseye.layers.cudnn.CuDNN.*;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CONVOLUTION;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Convolution layer.
 */
public class SimpleConvolutionLayer extends NNLayer implements LayerPrecision<SimpleConvolutionLayer> {
  
  
  /**
   * The Filter.
   */
  public final Tensor kernel;
  private int strideX = 1;
  private int strideY = 1;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json the json
   */
  protected SimpleConvolutionLayer(JsonObject json) {
    super(json);
    this.kernel = Tensor.fromJson(json.getAsJsonObject("filter"));
    this.strideX = json.get("strideX").getAsInt();
    this.strideY = json.get("strideY").getAsInt();
  }
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected SimpleConvolutionLayer() {
    this((Tensor) null);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param kernel the filter
   */
  protected SimpleConvolutionLayer(Tensor kernel) {
    super();
    if (kernel.getDimensions().length != 3) throw new IllegalArgumentException();
    if (kernel.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.kernel = kernel;
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   */
  public SimpleConvolutionLayer(final int width, int height, final int bands) {
    this(new Tensor(width, height, bands));
    assert (!false || 0 == (width - 1) % 2) : "Simple kernels must have odd width";
    assert (!false || 0 == (height - 1) % 2) : "Simple kernels must have odd height";
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
    json.add("filter", kernel.getJson());
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.addProperty("simple", false);
    return json;
  }
  
  /**
   * Add weights convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  public SimpleConvolutionLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.kernel.getData());
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    ((CudaExecutionContext) nncontext).initThread();
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputSize = batch.get(0).getDimensions();
    int[] kernelSize = this.kernel.getDimensions();
    int[] outputSize = getOutputSize(inputSize, kernelSize);
    int length = batch.length();
    
    try {
      
      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CudaResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
        precision.code, CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
      CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
        precision.code, CUDNN_TENSOR_NCHW, length, outputSize[2], outputSize[1], outputSize[0]);
      int paddingX = ((kernelSize[0] - 1) / 2);
      int paddingY = ((kernelSize[1] - 1) / 2);
      CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionNdDescriptor(CUDNN_CONVOLUTION, precision.code,
        new int[]{paddingX, paddingY},
        new int[]{strideY, strideX},
        new int[]{1, 1});
      CudaPtr alpha = CudaPtr.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, 1.0);
      CudaPtr beta = CudaPtr.javaPtr(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, 0.0);
      
      final double[] filterData = this.kernel.getData();
      CudaPtr filterPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, filterData);
      assert (0 < filterData.length);
      CudaPtr inputData = CudaPtr.toDevice(((CudaExecutionContext) nncontext).getDeviceNumber(), batch, precision);
      assert kernelSize[0] * kernelSize[1] * kernelSize[2] == filterData.length;
      
      CudaPtr outputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), Tensor.dim(outputSize) * 1l * length * precision.size);
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
        throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputSize)), e);
      }
      TensorList output = CudaPtr.fromDevice(outputBuffer, length, outputSize, cudnnHandle, precision);
      
      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
          ((CudaExecutionContext) nncontext).initThread();
          assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          int length = error.length();
          CudaPtr errorPtr = CudaPtr.toDevice(((CudaExecutionContext) nncontext).getDeviceNumber(), error, precision);
          if (!isFrozen()) {
            CudaPtr filterBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), filterData.length * 1l * precision.size);
            try {
              int backwardAlgorithm = ((CudaExecutionContext) nncontext).getBackwardFilterAlgorithm(
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
              CudaPtr workSpace = ((CudaExecutionContext) nncontext).allocateBackwardFilterWorkspace(((CudaExecutionContext) nncontext).getDeviceNumber(),
                inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), backwardAlgorithm);
              CuDNN.handle(cudnnConvolutionBackwardFilter(cudnnHandle, alpha.getPtr(),
                inputDescriptor.getPtr(), inputData.getPtr(),
                outputDescriptor.getPtr(), errorPtr.getPtr(),
                convolutionDescriptor.getPtr(), backwardAlgorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                filterDescriptor.getPtr(), filterBuffer.getPtr()));
              workSpace.finalize();
            } catch (Throwable e) {
              throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputSize)), e);
            }
            final Tensor weightGradient = CudaPtr.fromDevice(filterBuffer, precision, SimpleConvolutionLayer.this.kernel.getDimensions());
            buffer.get(SimpleConvolutionLayer.this, SimpleConvolutionLayer.this.kernel.getData()).addInPlace(weightGradient.getData());
          }
          if (input.isAlive()) {
            CudaPtr inputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(), batch.get(0).dim() * 1l * length * precision.size);
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
              throw new ComponentException(String.format("Error in convolution %s x %s => %s", Arrays.toString(inputSize), Arrays.toString(kernelSize), Arrays.toString(outputSize)), e);
            }
            TensorList inputBufferTensors = CudaPtr.fromDevice(inputBuffer, length, inputSize, cudnnHandle, precision);
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
    boolean cmp = cmp(outputSize, outputDims);
    return cmp;
  }
  
  private boolean cmp(int[] outputSize, int[] outputDims) {
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
  public SimpleConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.kernel.coordStream().parallel().forEach(c -> {
      this.kernel.set(c, f.applyAsDouble(c));
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
    this.kernel.coordStream().parallel().forEach(c -> {
      this.kernel.set(c, f.getAsDouble());
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.kernel.getData());
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
  public SimpleConvolutionLayer setStrideX(int strideX) {
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
  public SimpleConvolutionLayer setStrideY(int strideY) {
    this.strideY = strideY;
    return this;
  }
  
  public Precision getPrecision() {
    return precision;
  }
  
  public SimpleConvolutionLayer setPrecision(Precision precision) {
    this.precision = precision;
    return this;
  }
}
