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
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
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

public class DirectConvolutionLayer extends DirectCuDNNLayer {


  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("filter", filter.getJson());
    json.add("skip", skip.getJson());
    json.addProperty("simple", simple);
    return json;
  }

  public static DirectConvolutionLayer fromJson(JsonObject json) {
    return new DirectConvolutionLayer(json);
  }
  protected DirectConvolutionLayer(JsonObject json) {
    super(json);
    this.filter = Tensor.fromJson(json.getAsJsonObject("filter"));
    this.skip = Tensor.fromJson(json.getAsJsonObject("skip"));
    this.simple = json.getAsJsonPrimitive("simple").getAsBoolean();
  }


  public final Tensor filter;
  public final Tensor skip;
  public final boolean simple;

  protected DirectConvolutionLayer() {
    this((Tensor)null, (Tensor)null, true);
  }

  protected DirectConvolutionLayer(Tensor filter, Tensor skip, boolean simple) {
    super();
    this.simple = simple;
    this.skip = skip;
    if(filter.getDimensions().length != 3) throw new IllegalArgumentException();
    if(filter.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if(filter.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if(filter.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.filter = filter;
  }

  public DirectConvolutionLayer(final int width, int height, final int inputBands, final int outputBands) {
    this(width, height, inputBands * outputBands);
  }

  public DirectConvolutionLayer(final int width, int height, final int bands, boolean simple) {
    this(new Tensor(width,height,bands), new Tensor(new int[]{1,1}), simple);
    assert(!simple || 0 == (width-1) % 2) : "Simple kernels must have odd width";
    assert(!simple || 0 == (height-1) % 2) : "Simple kernels must have odd height";
  }

  public DirectConvolutionLayer(final int width, int height, final int bands) {
    this(width, height, bands, true);
  }

  public DirectConvolutionLayer(final int width, int height, final int inputBands, final int outputBands, boolean simple) {
    this(width, height, inputBands * outputBands, simple);
  }
  
  public DirectConvolutionLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.filter.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    final NNResult input = inObj[0];
    final TensorList batch = input.data;
    final int[] inputSize = batch.get(0).getDimensions();
    int[] kernelSize = this.filter.getDimensions();
    int[] outputSize = getOutputSize(inputSize, kernelSize);
    int length = batch.length();

    try {

      CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
              CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, outputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
      CuDNN.CuDNNResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, outputSize[2], outputSize[1], outputSize[0]);
      CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionDescriptor(
              simple ?((kernelSize[1] - 1) / 2):0, simple ?((kernelSize[0] - 1) / 2):0,
              1, 1,
              CUDNN_CONVOLUTION);
      CuDNN.CuDNNPtr alpha = CuDNN.javaPtr(1.0);
      CuDNN.CuDNNPtr beta = CuDNN.javaPtr(0.0);

      final double[] filterData = this.filter.getData();
      CuDNN.CuDNNPtr filterPtr = CuDNN.write(filterData);
      assert(0 < filterData.length);
      CuDNN.CuDNNPtr inputData = toDevice(batch);
      assert kernelSize[0] * kernelSize[1] * kernelSize[2] == filterData.length;

      CuDNN.CuDNNPtr outputBuffer = CuDNN.alloc(Tensor.dim(outputSize) * length * Sizeof.DOUBLE);
      CuDNN.devicePool.with(device -> {
        try {
          assert verifyOutputDims(inputDescriptor, filterDescriptor, convolutionDescriptor, outputSize);
          int algorithm = device.getForwardAlgorithm(
                  inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
          CuDNN.CuDNNPtr workSpace = device.allocateForwardWorkspace(
                  inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
          CuDNN.handle(cudnnConvolutionForward(device.cudnnHandle, alpha.getPtr(),
                  inputDescriptor.getPtr(), inputData.getPtr(),
                  filterDescriptor.getPtr(), filterPtr.getPtr(),
                  convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                  outputDescriptor.getPtr(), outputBuffer.getPtr()));
          workSpace.finalize();
        } catch (Throwable e) {
          throw new RuntimeException("Error with " + Arrays.toString(kernelSize),e);
        }
      });
      TensorList output = fromDevice(outputBuffer, length, outputSize);

      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          assert (error.length() == batch.length());
          //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          int length = error.length();
          CuDNN.CuDNNPtr errorPtr = toDevice(error);
          if (!isFrozen()) {
            CuDNN.CuDNNPtr filterBuffer = CuDNN.alloc(filterData.length * Sizeof.DOUBLE);
            try {
              CuDNN.devicePool.with(device -> {
                int algorithm = device.getBackwardFilterAlgorithm(
                        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
                CuDNN.CuDNNPtr workSpace = device.allocateBackwardFilterWorkspace(
                        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
                CuDNN.handle(cudnnConvolutionBackwardFilter(device.cudnnHandle, alpha.getPtr(),
                        inputDescriptor.getPtr(), inputData.getPtr(),
                        outputDescriptor.getPtr(), errorPtr.getPtr(),
                        convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                        filterDescriptor.getPtr(), filterBuffer.getPtr()));
                workSpace.finalize();
              });
            } catch (Throwable e) {
              throw new RuntimeException("Error with " + Arrays.toString(kernelSize),e);
            }
            final Tensor weightGradient = fromDevice(filterBuffer, DirectConvolutionLayer.this.filter.getDimensions());
            buffer.get(DirectConvolutionLayer.this, DirectConvolutionLayer.this.filter).accumulate(weightGradient.getData());
          }
          if (input.isAlive()) {
            CuDNN.CuDNNPtr inputBuffer = CuDNN.alloc(batch.get(0).dim() * length * Sizeof.DOUBLE);
            try {
              CuDNN.devicePool.with(device -> {
                int algorithm = device.getBackwardDataAlgorithm(
                        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
                CuDNN.CuDNNPtr workSpace = device.allocateBackwardDataWorkspace(
                        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
                CuDNN.handle(cudnnConvolutionBackwardData(device.cudnnHandle, alpha.getPtr(),
                        filterDescriptor.getPtr(), filterPtr.getPtr(),
                        outputDescriptor.getPtr(), errorPtr.getPtr(),
                        convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                        inputDescriptor.getPtr(), inputBuffer.getPtr()));
              });
            } catch (Throwable e) {
              throw new RuntimeException("Error with " + Arrays.toString(kernelSize),e);
            }
            TensorList inputBufferTensors = fromDevice(inputBuffer, length, inputSize);
            input.accumulate(buffer, inputBufferTensors);
          }
        }

        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new RuntimeException("Error with image res " + Arrays.toString(inputSize),e);
    }
  }

  protected int[] getOutputSize(int[] inputSize, int[] kernelSize) {
    return IntStream.range(0, kernelSize.length).map(i -> {
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
  }


  protected boolean verifyOutputDims(CuDNN.CuDNNResource<cudnnTensorDescriptor> inputDescriptor, CuDNN.CuDNNResource<cudnnFilterDescriptor> filterDescriptor, CuDNN.CuDNNResource<cudnnConvolutionDescriptor> convolutionDescriptor, int[] outputSize) {
    int[] outputDims = CuDNN.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr());
    if(4 != outputDims.length) return false;
    if(outputSize[0] != outputDims[3]) return false;
    if(outputSize[1] != outputDims[2]) return false;
    if(outputSize[2] != outputDims[1]) return false;
    return true;
  }

  public DirectConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.filter.coordStream().parallel().forEach(c -> {
      this.filter.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  public DirectConvolutionLayer setWeights(final DoubleSupplier f) {
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
