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
import com.simiacryptus.mindseye.lang.*;
import jcuda.jcudnn.cudnnTensorDescriptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Reduces the resolution of the input by selecting a centered window. The output image will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgCropLayer extends NNLayer implements LayerPrecision<ImgCropLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgCropLayer.class);
  
  private int sizeX;
  private int sizeY;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   */
  private ImgCropLayer() {
  }
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgCropLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgCropLayer(final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static ImgCropLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ImgCropLayer(json, rs);
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public NNLayer getCompatibilityLayer() {
    return this.as(com.simiacryptus.mindseye.layers.java.ImgCropLayer.class);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    if (((CudaExecutionContext) nncontext).getDeviceNumber() < 0)
      return getCompatibilityLayer().eval(nncontext, inObj);
    ((CudaExecutionContext) nncontext).initThread();
    assert 1 == inObj.length;
    assert 3 == inObj[0].getData().getDimensions().length;
    final int length = inObj[0].getData().length();
    int[] dimIn = inObj[0].getData().getDimensions();
    final int[] dimOut = Arrays.copyOf(dimIn, 3);
    dimOut[0] = sizeX;
    dimOut[1] = sizeY;
    final CudaPtr inputBuffer = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, inObj[0].getData());
    final CudaPtr outputBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(),
                                             length * dimOut[2] * dimOut[1] * dimOut[0] * precision.size);
    copy((CuDNN) nncontext, length, dimIn, inputBuffer, dimOut, outputBuffer);
    final TensorList outputData = new GpuTensorList(outputBuffer, length, dimOut, ((CuDNN) nncontext).cudnnHandle, precision);
    return new NNResult(outputData) {
  
      @Override
      public void free() {
        inputBuffer.finalize();
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
        if (!Arrays.equals(error.getDimensions(), outputData.getDimensions())) {
          throw new AssertionError(Arrays.toString(error.getDimensions()) + " != " + Arrays.toString(outputData.getDimensions()));
        }
        if (error.length() != outputData.length()) {
          throw new AssertionError(error.length() + " != " + outputData.length());
        }
  
        assert error.length() == inObj[0].getData().length();
        if (inObj[0].isAlive()) {
          ((CudaExecutionContext) nncontext).initThread();
          final CudaPtr errorPtr = CudaPtr.write(((CudaExecutionContext) nncontext).getDeviceNumber(), precision, error);
          final CudaPtr passbackBuffer = CuDNN.alloc(((CudaExecutionContext) nncontext).getDeviceNumber(),
                                                     length * dimIn[2] * dimIn[1] * dimIn[0] * precision.size);
          copy((CuDNN) nncontext, length, dimOut, errorPtr, dimIn, passbackBuffer);
          final TensorList passbackTensorList = new GpuTensorList(passbackBuffer, length, dimIn, ((CuDNN) nncontext).cudnnHandle, precision);
          inObj[0].accumulate(buffer, passbackTensorList);
          passbackBuffer.finalize();
          errorPtr.finalize();
        }
        free();
      }
  
      @Override
      public boolean isAlive() {
        return Arrays.stream(inObj).anyMatch(x -> x.isAlive());
      }
    };
  }
  
  /**
   * Copy.
   *
   * @param nncontext   the nncontext
   * @param length      the length
   * @param dimIn       the dim in
   * @param source      the input buffer
   * @param dimOut      the dim out
   * @param destination the output buffer
   */
  public void copy(CuDNN nncontext, int length, int[] dimIn, CudaPtr source, int[] dimOut, CudaPtr destination) {
    if (source.getDeviceId() != destination.getDeviceId()) throw new IllegalArgumentException("device id mismatch");
    if (source.getDeviceId() != nncontext.getDeviceNumber())
      throw new IllegalArgumentException(String.format("device id mismatch: %d != %d", source.getDeviceId(), nncontext.getDeviceNumber()));
    if (3 != dimIn.length) throw new IllegalArgumentException("dimIn.length");
    if (3 != dimOut.length) throw new IllegalArgumentException("dimOut.length");
    final double offsetX = ((dimOut[0] - dimIn[0]) / 2.0);
    final double offsetY = ((dimOut[1] - dimIn[1]) / 2.0);
    //log.info(String.format("offset=%d,%d", offsetX, offsetY));
    final int[] viewDim = new int[3];
    Arrays.parallelSetAll(viewDim, i -> Math.min(dimIn[i], dimOut[i]));
    final CudaResource<cudnnTensorDescriptor> sourceViewDescriptor = CuDNN.newTensorDescriptor(
      precision.code,
      length,
      viewDim[2],
      viewDim[1],
      viewDim[0],
      dimIn[2] * dimIn[1] * dimIn[0],
      dimIn[1] * dimIn[0],
      dimIn[0],
      1);
    final CudaResource<cudnnTensorDescriptor> destinationViewDescriptor = CuDNN.newTensorDescriptor(
      precision.code,
      length,
      viewDim[2],
      viewDim[1],
      viewDim[0],
      dimOut[2] * dimOut[1] * dimOut[0],
      dimOut[1] * dimOut[0],
      dimOut[0],
      1);
    int inOffset = 0;
    int outputOffset = 0;
    if (offsetX < 0) {
      inOffset += Math.floor(-offsetX) * precision.size;
    }
    else {
      outputOffset += Math.floor(offsetX) * precision.size;
    }
    if (offsetY < 0) {
      inOffset += Math.floor(-offsetY) * dimIn[0] * precision.size;
    }
    else {
      outputOffset += Math.floor(offsetY) * dimOut[0] * precision.size;
    }
    CuDNN.cudnnTransformTensor(nncontext.cudnnHandle,
                               precision.getPointer(1.0),
                               sourceViewDescriptor.getPtr(), source.getPtr().withByteOffset(inOffset),
                               precision.getPointer(0.0),
                               destinationViewDescriptor.getPtr(), destination.getPtr().withByteOffset(outputOffset)
                              );
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.addProperty("sizeY", sizeY);
    json.addProperty("sizeX", sizeX);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public ImgCropLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
}
