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
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaDevice;
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudnnHandle;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Reduces the resolution of the input by selecting a centered window. The output png will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgTileCycleLayer extends LayerBase implements MultiPrecision<ImgTileCycleLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgTileCycleLayer.class);
  private double xPos = 0.5;
  private double yPos = 0.5;
  
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img eval layer.
   */
  public ImgTileCycleLayer() {
  }
  
  /**
   * Instantiates a new Img eval layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgTileCycleLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json img eval layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img eval layer
   */
  public static ImgTileCycleLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileCycleLayer(json, rs);
  }
  
  /**
   * Copy cuda tensor.
   *
   * @param gpu       the gpu
   * @param input     the input tensor
   * @param length    the length
   * @param precision the precision
   * @param splitX    the split x
   * @param splitY    the split y
   * @return the cuda tensor
   */
  public static CudaTensor copy(final CudnnHandle gpu, final CudaTensor input, final int length, Precision precision, final int splitX, final int splitY) {
    CudaMemory inputTensorMemory = input.getMemory(gpu);
    try {
      @Nonnull final CudaDevice.CudaTensorDescriptor imageDescriptor = gpu.newTensorDescriptor(
        precision,//
        length,//
        input.descriptor.channels,//
        input.descriptor.height,//
        input.descriptor.width,//
        input.descriptor.nStride,//
        input.descriptor.cStride,//
        input.descriptor.hStride,//
        input.descriptor.wStride);
      @Nonnull final CudaMemory outputBuffer = gpu.allocate((long) length * imageDescriptor.nStride * precision.size, MemoryType.Managed.normalize(), true);
  
  
      int splitY1 = splitY;
      int splitY2 = input.descriptor.height - splitY1;
      int splitX1 = splitX;
      int splitX2 = input.descriptor.width - splitX1;
  
      {
        @Nonnull final CudaDevice.CudaTensorDescriptor tileDescriptor = gpu.newTensorDescriptor(
          precision,//
          length,//
          input.descriptor.channels,//
          splitY1,//
          splitX1,//
          input.descriptor.nStride,//
          input.descriptor.cStride,//
          input.descriptor.hStride,//
          input.descriptor.wStride);
        try {
          CudaSystem.handle(gpu.cudnnTransformTensor(
            precision.getPointer(1.0),
            tileDescriptor.getPtr(), inputTensorMemory.getPtr().withByteOffset(0 * precision.size),
            precision.getPointer(0.0),
            tileDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset((splitY2 * input.descriptor.hStride + splitX2 * input.descriptor.wStride) * precision.size)
          ));
        } finally {
          tileDescriptor.freeRef();
        }
      }
  
      {
        @Nonnull final CudaDevice.CudaTensorDescriptor tileDescriptor = gpu.newTensorDescriptor(
          precision,//
          length,//
          input.descriptor.channels,//
          splitY2,//
          splitX1,//
          input.descriptor.nStride,//
          input.descriptor.cStride,//
          input.descriptor.hStride,//
          input.descriptor.wStride);
        try {
          CudaSystem.handle(gpu.cudnnTransformTensor(
            precision.getPointer(1.0),
            tileDescriptor.getPtr(), inputTensorMemory.getPtr().withByteOffset(splitY1 * input.descriptor.hStride * precision.size),
            precision.getPointer(0.0),
            tileDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(splitX2 * input.descriptor.wStride * precision.size)
          ));
        } finally {
          tileDescriptor.freeRef();
        }
      }
  
      {
        @Nonnull final CudaDevice.CudaTensorDescriptor tileDescriptor = gpu.newTensorDescriptor(
          precision,//
          length,//
          input.descriptor.channels,//
          splitY1,//
          splitX2,//
          input.descriptor.nStride,//
          input.descriptor.cStride,//
          input.descriptor.hStride,//
          input.descriptor.wStride);
        try {
          CudaSystem.handle(gpu.cudnnTransformTensor(
            precision.getPointer(1.0),
            tileDescriptor.getPtr(), inputTensorMemory.getPtr().withByteOffset(splitX1 * input.descriptor.wStride * precision.size),
            precision.getPointer(0.0),
            tileDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(splitY2 * input.descriptor.hStride * precision.size)
          ));
        } finally {
          tileDescriptor.freeRef();
        }
      }
  
      @Nonnull final CudaDevice.CudaTensorDescriptor tileDescriptor = gpu.newTensorDescriptor(
        precision,//
        length,//
        input.descriptor.channels,//
        splitY2,//
        splitX2,//
        input.descriptor.nStride,//
        input.descriptor.cStride,//
        input.descriptor.hStride,//
        input.descriptor.wStride);
      try {
        CudaSystem.handle(gpu.cudnnTransformTensor(
          precision.getPointer(1.0),
          tileDescriptor.getPtr(), inputTensorMemory.getPtr().withByteOffset((splitY1 * input.descriptor.hStride + splitX1 * input.descriptor.wStride) * precision.size),
          precision.getPointer(0.0),
          tileDescriptor.getPtr(), outputBuffer.getPtr().withByteOffset(0 * precision.size)
        ));
      } finally {
        tileDescriptor.freeRef();
      }
  
  
      inputTensorMemory.dirty();
      outputBuffer.dirty();
      return CudaTensor.wrap(outputBuffer, imageDescriptor, precision);
    } finally {
      inputTensorMemory.freeRef();
    }
  }
  
  /**
   * Get view dimensions int [ ].
   *
   * @param sourceDimensions      the source dimensions
   * @param destinationDimensions the destination dimensions
   * @return the int [ ]
   */
  @Nonnull
  public static int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions) {
    @Nonnull final int[] viewDim = new int[3];
    Arrays.parallelSetAll(viewDim, i -> Math.min(sourceDimensions[i], destinationDimensions[i]));
    return viewDim;
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  @Nonnull
  public Layer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
  }
  
  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) return getCompatibilityLayer().evalAndFree(inObj);
    assert 1 == inObj.length;
    final Result input = inObj[0];
    final TensorList inputData = input.getData();
    assert 3 == inputData.getDimensions().length;
    final int length = inputData.length();
    @Nonnull int[] dimIn = inputData.getDimensions();
    int splitX1 = (int) (dimIn[0] * getxPos());
    int splitX2 = dimIn[0] - splitX1;
    int splitY1 = (int) (dimIn[1] * getyPos());
    int splitY2 = dimIn[1] - splitY1;
    final TensorList outputData = CudaSystem.run(gpu -> {
      @Nullable final CudaTensor inputTensor = gpu.getTensor(inputData, precision, MemoryType.Device, false);
      inputData.freeRef();
      CudaTensor cudaTensor = copy(gpu, inputTensor, length, precision, splitX1, splitY1);
      inputTensor.freeRef();
      return CudaTensorList.wrap(cudaTensor, length, dimIn, precision);
    }, inputData);
    return new Result(outputData, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (!Arrays.equals(delta.getDimensions(), outputData.getDimensions())) {
        throw new AssertionError(Arrays.toString(delta.getDimensions()) + " != " + Arrays.toString(outputData.getDimensions()));
      }
      if (delta.length() != outputData.length()) {
        throw new AssertionError(delta.length() + " != " + outputData.length());
      }
      assert delta.length() == length;
      if (input.isAlive()) {
        final TensorList passbackTensorList = CudaSystem.run(gpu -> {
          @Nullable final CudaTensor errorPtr = gpu.getTensor(delta, precision, MemoryType.Device, false);
          delta.freeRef();
          CudaTensor cudaTensor = copy(gpu, errorPtr, length, precision, splitX2, splitY2);
          errorPtr.freeRef();
          return CudaTensorList.wrap(cudaTensor, length, dimIn, precision);
        }, delta);
        input.accumulate(buffer, passbackTensorList);
      }
      else {
        delta.freeRef();
      }
      
      
    }) {
      
      @Override
      public void accumulate(final DeltaSet<Layer> buffer, final TensorList delta) {
        getAccumulator().accept(buffer, delta);
      }
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
      }
    };
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public ImgTileCycleLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  public double getxPos() {
    return xPos;
  }
  
  public double getyPos() {
    return yPos;
  }
  
  public ImgTileCycleLayer setXPos(double xPos) {
    this.xPos = xPos;
    return this;
  }
  
  public ImgTileCycleLayer setYPos(double yPos) {
    this.yPos = yPos;
    return this;
  }
}
