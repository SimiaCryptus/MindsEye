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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaDevice;
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory;
import com.simiacryptus.mindseye.lang.cudnn.CudaResource;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import jcuda.jcudnn.cudnnIndicesType;
import jcuda.jcudnn.cudnnNanPropagation;
import jcuda.jcudnn.cudnnReduceTensorDescriptor;
import jcuda.jcudnn.cudnnReduceTensorIndices;
import jcuda.jcudnn.cudnnReduceTensorOp;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Similar to the pooling layer, but the pool size is always the image size. The output dimensions are always 1x1xN.
 */
@SuppressWarnings("serial")
public class BandAvgReducerLayer extends LayerBase implements MultiPrecision<BandAvgReducerLayer> {
  
  private Precision precision = Precision.Double;
  private double alpha = 1.0;
  
  /**
   * Instantiates a new Pooling layer.
   */
  public BandAvgReducerLayer() {
    super();
  }
  
  /**
   * Instantiates a new Pooling layer.
   *
   * @param json the json
   */
  protected BandAvgReducerLayer(@Nonnull final JsonObject json) {
    super(json);
    precision = Precision.valueOf(json.get("precision").getAsString());
    alpha = json.get("alpha").getAsDouble();
  }
  
  /**
   * From json pooling layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the pooling layer
   */
  public static BandAvgReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BandAvgReducerLayer(json);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    throw new RuntimeException("Not Implemented");
  }
  
  @Nullable
  @Override
  public Result evalAndFree(final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    final Result input = inObj[0];
    TensorList inputData = input.getData();
    @Nonnull final int[] inputSize = inputData.getDimensions();
    int length = inputData.length();
    
    final int bands = inputSize[2];
    CudaTensorList result = CudaSystem.run(gpu -> {
      CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
      @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision, length, bands, 1, 1);
      long size = (long) precision.size * outputDescriptor.nStride * length;
      @Nonnull final CudaMemory outputPtr = gpu.allocate(size, MemoryType.Managed, true);
      CudaResource<cudnnReduceTensorDescriptor> reduceTensorDescriptor = gpu.cudnnCreateReduceTensorDescriptor(
        cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_AVG, precision.code, cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
        cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES, cudnnIndicesType.CUDNN_32BIT_INDICES);
      
      CudaMemory inputMemory = inputTensor.getMemory(gpu);
      @Nonnull final CudaMemory workspacePtr = gpu.allocate(inputMemory.size, MemoryType.Device, true);
      @Nonnull final CudaMemory indexPtr = gpu.allocate(12 * length, MemoryType.Device, false);
      
      gpu.cudnnReduceTensor(reduceTensorDescriptor.getPtr(),
        indexPtr.getPtr(), indexPtr.size, workspacePtr.getPtr(), workspacePtr.size,
        precision.getPointer(alpha), inputTensor.descriptor.getPtr(), inputMemory.getPtr(),
        precision.getPointer(0.0), outputDescriptor.getPtr(), outputPtr.getPtr());
      outputPtr.dirty();
      inputMemory.dirty();
  
      Stream.of(inputMemory, inputTensor, reduceTensorDescriptor, workspacePtr, indexPtr, inputData).forEach(ReferenceCounting::freeRef);
      return CudaTensorList.wrap(CudaTensor.wrap(outputPtr, outputDescriptor, precision), length, new int[]{1, 1, bands}, precision);
    });
    int pixels = inputSize[0] * inputSize[1];
    return new Result(result, (DeltaSet<Layer> ctx, TensorList delta) -> {
      TensorList passback;
      passback = TensorArray.wrap(delta.stream().map(x -> {
        Tensor tensor = new Tensor(inputSize[0], inputSize[1], inputSize[2])
          .setByCoord(c -> x.get(c.getCoords()[2]) * alpha / pixels);
        x.freeRef();
        return tensor;
      }).toArray(i -> new Tensor[i]));
//      passback = CudaSystem.generate(gpu -> {
//        CudaTensor deltaTensor = gpu.getTensor(delta, precision, MemoryType.Device, true);
//        @Nonnull final CudaDevice.CudaTensorDescriptor outputDescriptor = gpu.newTensorDescriptor(precision,
//          length, inputSize[2], inputSize[1], inputSize[0]);
//        @Nonnull final CudaMemory outputPtr = gpu.allocate((long) precision.size * outputDescriptor.nStride * length, MemoryType.Device, true);
//        CudaMemory deltaMemory = deltaTensor.getMemory(gpu);
//        @Nonnull final CudaDevice.CudaTensorDescriptor inputDescriptor = gpu.newTensorDescriptor(precision,
//          1, 1, inputSize[1], inputSize[0]);
//        for(int batch=0;batch<length;batch++){
//          Tensor tensor = delta.get(batch);
//          for(int band=0;band<bands;band++){
//            int i = batch * bands + band;
//            CudaMemory img = outputPtr.withByteOffset(precision.size * i * outputDescriptor.cStride);
//            CudaMemory val = deltaMemory.withByteOffset(precision.size * i);
//            gpu.cudnnSetTensor(inputDescriptor.getPtr(), img.getPtr(), precision.getPointer(tensor.get(band) / outputDescriptor.cStride));
//            img.freeRef();
//            val.freeRef();
//            outputPtr.dirty().synchronize();
//          }
//        }
//        Stream.of(deltaMemory, deltaTensor, inputDescriptor).forEach(ReferenceCounting::freeRef);
//        return CudaTensorList.wrap(CudaTensor.wrap(outputPtr, outputDescriptor, precision), length, inputSize, precision);
//      });
      input.accumulate(ctx, passback);
    }) {
      @Override
      protected void _free() {
        super._free();
        input.freeRef();
      }
    };
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("alpha", alpha);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public BandAvgReducerLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  /**
   * Gets alpha.
   *
   * @return the alpha
   */
  public double getAlpha() {
    return alpha;
  }
  
  /**
   * Sets alpha.
   *
   * @param alpha the alpha
   * @return the alpha
   */
  public BandAvgReducerLayer setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }
}
