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
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.cudnn.CudaPtr;
import com.simiacryptus.mindseye.layers.cudnn.CudaResource;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.*;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CONVOLUTION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Convolution layer.
 */
public class ConvolutionLayer extends NNLayer {

  private Map<Integer, GPUDataMirror> stateCache = new HashMap<>();
  
  public static class CudnnFloatDelta extends DeltaBuffer {
    public CudnnFloatDelta(double[] values, NNLayer layer) {
      super(values, null, layer);
    }
    
    CudaPtr buffer;
 
    public void accumulate(CudaResource<cudnnTensorDescriptor> size, CudaPtr data) {
      if(null != buffer) {
        CuDNN.devicePool.with(handle->{
          CuDNN.handle(cudnnAddTensor(handle.cudnnHandle,
            Pointer.to(new float[]{1.0f}), size.getPtr(), data.getPtr(),
            Pointer.to(new float[]{1.0f}), size.getPtr(), buffer.getPtr()));
        });
        data.finalize();
      } else {
        buffer = data;
      }
    }
    
    @Override
    public double[] getDelta() {
      if(null == delta) {
        float[] data = new float[length()];
        buffer.read(data);
        this.delta = Tensor.toDoubles(data);
      }
      return super.getDelta();
    }
  }
  
  public static class GPUDataMirror {
    private long fingerprint;
    private int[] indicies;
    private volatile CudaPtr ptr;
  
    public GPUDataMirror(int length) {
      this.indicies = IntStream.range(0, 3).map(i -> new Random().nextInt(length)).distinct().limit(3).toArray();
    }
  
    public CudaPtr upload(int device, float[] data) {
      long inputHash = hashFunction(data);
      if(null != ptr && inputHash == fingerprint) return ptr;
      this.fingerprint = inputHash;
      return ptr = CuDNN.write(device, data);
    }
  
    public CudaPtr upload(int device, double[] data) {
      long inputHash = hashFunction(data);
      if(null != ptr && inputHash == fingerprint) return ptr;
      this.fingerprint = inputHash;
      return ptr = CuDNN.write(device, data);
    }
  
    public CudaPtr uploadAsFloats(int device, double[] data) {
      long inputHash = hashFunction(data);
      if(null != ptr && inputHash == fingerprint) return ptr;
      this.fingerprint = inputHash;
      return ptr = CuDNN.write(device, Tensor.toFloats(data));
    }
  
    public long hashFunction(float[] data) {
      return IntStream.of(indicies).mapToObj(i->data[i])
               .mapToInt(Float::floatToIntBits)
               .reduce((a,b)->a^b).getAsInt();
    }
    public long hashFunction(double[] data) {
      return IntStream.of(indicies).mapToDouble(i->data[i])
               .mapToLong(Double::doubleToLongBits)
               .reduce((a,b)->a^b).getAsLong();
    }
  }

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
    this((Tensor)null, (Tensor)null, true);
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
    if(filter.getDimensions().length != 3) throw new IllegalArgumentException();
    if(filter.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if(filter.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if(filter.getDimensions()[2] <= 0) throw new IllegalArgumentException();
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
    this(new Tensor(width,height,bands), new Tensor(new int[]{1,1}), simple);
    assert(!simple || 0 == (width-1) % 2) : "Simple kernels must have odd width";
    assert(!simple || 0 == (height-1) % 2) : "Simple kernels must have odd height";
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
    CuDNN.setDevice(nncontext.getCudaDeviceId());
    final NNResult input = inObj[0];
    final TensorList batch = input.data;
    final int[] inputSize = batch.getDimensions();
    int[] kernelSize = this.filter.getDimensions();
    int length = batch.length();

    try {

      CudaResource<cudnnTensorDescriptor> inputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, inputSize[2], inputSize[1], inputSize[0]);
      CudaResource<cudnnFilterDescriptor> filterDescriptor = CuDNN.newFilterDescriptor(
              CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernelSize[2] / inputSize[2], inputSize[2], kernelSize[1], kernelSize[0]);
      CudaResource<cudnnConvolutionDescriptor> convolutionDescriptor = CuDNN.newConvolutionDescriptor(
          simple ?((kernelSize[1] - 1) / 2):0, simple ?((kernelSize[0] - 1) / 2):0,
          strideX, strideY, CUDNN_CONVOLUTION);
      int[] outputDims = CuDNN.getOutputDims(inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr());
      int[] outputSize = new int[]{outputDims[3], outputDims[2], outputDims[1]};
      CudaResource<cudnnTensorDescriptor> outputDescriptor = CuDNN.newTensorDescriptor(
              CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, outputSize[2], outputSize[1], outputSize[0]);
      CudaPtr alpha = CuDNN.javaPtr(nncontext.getCudaDeviceId(), 1.0f);
      CudaPtr beta = CuDNN.javaPtr(nncontext.getCudaDeviceId(), 0.0f);

      CudaPtr filterPtr = stateCache.computeIfAbsent(nncontext.getCudaDeviceId(),i->new GPUDataMirror(filter.dim()))
                            .uploadAsFloats(nncontext.getCudaDeviceId(), filter.getData());
      CudaPtr inputData = CudaPtr.toDeviceAsFloat(nncontext.getCudaDeviceId(), batch);
      CudaPtr outputBuffer = CuDNN.alloc(nncontext.getCudaDeviceId(), Tensor.dim(outputSize) * 1l * length * Sizeof.FLOAT);
      CuDNN.devicePool.with(device -> {
        try {
          int algorithm = device.getForwardAlgorithm(
                  inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
          CudaPtr workSpace = device.allocateForwardWorkspace(nncontext.getCudaDeviceId(),
            inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
          CuDNN.handle(cudnnConvolutionForward(device.cudnnHandle, alpha.getPtr(),
                  inputDescriptor.getPtr(), inputData.getPtr(),
                  filterDescriptor.getPtr(), filterPtr.getPtr(),
                  convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                  outputDescriptor.getPtr(), outputBuffer.getPtr()));
          workSpace.finalize();
        } catch (Throwable e) {
          throw new RuntimeException("Error map " + Arrays.toString(kernelSize),e);
        }
      });
      TensorList output = CudaPtr.fromDeviceFloat(outputBuffer, length, outputSize);

      return new NNResult(output) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList error) {
          outputBuffer.finalize();
          CuDNN.setDevice(nncontext.getCudaDeviceId());
          assert (error.length() == batch.length());
          int length = error.length();
          CudaPtr errorPtr = CudaPtr.toDeviceAsFloat(nncontext.getCudaDeviceId(), error);
          if (!isFrozen()) {
            CudaPtr filterBuffer = CuDNN.alloc(nncontext.getCudaDeviceId(), filter.dim() * 1l * Sizeof.FLOAT);
            try {
              CuDNN.devicePool.with(device -> {
                int algorithm = device.getBackwardFilterAlgorithm(
                        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
                CudaPtr workSpace = device.allocateBackwardFilterWorkspace(nncontext.getCudaDeviceId(),
                  inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
                CuDNN.handle(cudnnConvolutionBackwardFilter(device.cudnnHandle, alpha.getPtr(),
                        inputDescriptor.getPtr(), inputData.getPtr(),
                        outputDescriptor.getPtr(), errorPtr.getPtr(),
                        convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                        filterDescriptor.getPtr(), filterBuffer.getPtr()));
                workSpace.finalize();
              });
            } catch (Throwable e) {
              throw new RuntimeException("Error map " + Arrays.toString(kernelSize),e);
            }
            buffer.get(ConvolutionLayer.this, ()->new CudnnFloatDelta(filter.getData(),ConvolutionLayer.this))
              .accumulate(CuDNN.newTensorDescriptor(CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, kernelSize[2], kernelSize[1], kernelSize[0]), filterBuffer);
          }
          if (input.isAlive()) {
            CudaPtr inputBuffer = CuDNN.alloc(nncontext.getCudaDeviceId(), Tensor.dim(batch.getDimensions()) * 1l * length * Sizeof.FLOAT);
            try {
              CuDNN.devicePool.with(device -> {
                int algorithm = device.getBackwardDataAlgorithm(
                        inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr());
                CudaPtr workSpace = device.allocateBackwardDataWorkspace(nncontext.getCudaDeviceId(),
                  inputDescriptor.getPtr(), filterDescriptor.getPtr(), convolutionDescriptor.getPtr(), outputDescriptor.getPtr(), algorithm);
                CuDNN.handle(cudnnConvolutionBackwardData(device.cudnnHandle, alpha.getPtr(),
                        filterDescriptor.getPtr(), filterPtr.getPtr(),
                        outputDescriptor.getPtr(), errorPtr.getPtr(),
                        convolutionDescriptor.getPtr(), algorithm, workSpace.getPtr(), workSpace.size, beta.getPtr(),
                        inputDescriptor.getPtr(), inputBuffer.getPtr()));
                workSpace.finalize();
              });
            } catch (Throwable e) {
              throw new RuntimeException("Error map " + Arrays.toString(kernelSize),e);
            }
            TensorList inputBufferTensors = CudaPtr.fromDeviceFloat(inputBuffer, length, inputSize);
            input.accumulate(buffer, inputBufferTensors);
            inputBuffer.finalize();
          }
        }

        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }
      };
    } catch (Throwable e) {
      throw new RuntimeException("Error map image res " + Arrays.toString(inputSize),e);
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
  
  public int getStrideX() {
    return strideX;
  }
  
  public ConvolutionLayer setStrideX(int strideX) {
    this.strideX = strideX;
    return this;
  }
  
  public ConvolutionLayer setStrideXY(int strideX, int strideY) {
    this.strideX = strideX;
    this.strideY = strideY;
    return this;
  }
  
  public int getStrideY() {
    return strideY;
  }
  
  public ConvolutionLayer setStrideY(int strideY) {
    this.strideY = strideY;
    return this;
  }
}
