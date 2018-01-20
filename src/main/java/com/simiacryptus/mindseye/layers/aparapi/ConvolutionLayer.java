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

package com.simiacryptus.mindseye.layers.aparapi;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.Util;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * This convolution layer is often used as the reference implementation for other convolution implementation. It uses
 * OpenCL via Aparapi to compile Java into GPU-accellerated kernels. Due to its simple implementation and limitations of
 * Aparapi, it is not as fast as CuDNN-powered layers.
 */
@SuppressWarnings("serial")
public class ConvolutionLayer extends NNLayer {
  
  
  /**
   * The Kernel.
   */
  public final Tensor kernel;
  private Integer paddingX = null;
  private Integer paddingY = null;
  
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected ConvolutionLayer() {
    this(null, true);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   */
  public ConvolutionLayer(final int width, final int height, final int bands) {
    this(width, height, bands, true);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width  the width
   * @param height the height
   * @param bands  the bands
   * @param simple the simple
   */
  public ConvolutionLayer(final int width, final int height, final int bands, final boolean simple) {
    this(new Tensor(width, height, bands), simple);
    assert !simple || 0 == (width - 1) % 2 : "Simple kernels must have odd width";
    assert !simple || 0 == (height - 1) % 2 : "Simple kernels must have odd height";
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param width       the width
   * @param height      the height
   * @param inputBands  the input bands
   * @param outputBands the output bands
   */
  public ConvolutionLayer(final int width, final int height, final int inputBands, final int outputBands) {
    this(width, height, inputBands * outputBands);
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
  public ConvolutionLayer(final int width, final int height, final int inputBands, final int outputBands, final boolean simple) {
    this(width, height, inputBands * outputBands, simple);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ConvolutionLayer(final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    kernel = Tensor.fromJson(json.get("filter"), resources);
    JsonElement paddingX = json.get("paddingX");
    if (null != paddingX && paddingX.isJsonPrimitive()) this.setPaddingX((paddingX.getAsInt()));
    JsonElement paddingY = json.get("paddingY");
    if (null != paddingY && paddingY.isJsonPrimitive()) this.setPaddingY((paddingY.getAsInt()));
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param kernel the kernel
   * @param simple the simple
   */
  protected ConvolutionLayer(final Tensor kernel, final boolean simple) {
    super();
    this.paddingX = simple ? null : 0;
    this.paddingY = simple ? null : 0;
    int[] dimensions = kernel.getDimensions();
    if (dimensions.length != 3) throw new IllegalArgumentException(Arrays.toString(dimensions));
    if (dimensions[0] <= 0) throw new IllegalArgumentException(Arrays.toString(dimensions));
    if (dimensions[1] <= 0) throw new IllegalArgumentException(Arrays.toString(dimensions));
    if (dimensions[2] <= 0) throw new IllegalArgumentException(Arrays.toString(dimensions));
    if (dimensions[2] <= 0) throw new IllegalArgumentException(Arrays.toString(dimensions));
    this.kernel = kernel;
  }
  
  /**
   * From json convolution layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the convolution layer
   */
  public static ConvolutionLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ConvolutionLayer(json, rs);
  }
  
  /**
   * Add weights convolution layer.
   *
   * @param f the f
   * @return the convolution layer
   */
  public ConvolutionLayer addWeights(final DoubleSupplier f) {
    Util.add(f, kernel.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
    
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputDims = batch.get(0).getDimensions();
    final int[] kernelDims = kernel.getDimensions();
    final double[] kernelData = ConvolutionLayer.this.kernel.getData();
    final ConvolutionController convolutionController = new ConvolutionController(inputDims, kernelDims, paddingX, paddingY);
    final Tensor[] output = IntStream.range(0, batch.length())
                                     .mapToObj(dataIndex -> new Tensor(convolutionController.getOutputDims()))
                                     .toArray(i -> new Tensor[i]);
    try {
      final double[][] inputBuffers = batch.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
      final double[][] outputBuffers = Arrays.stream(output).map(x -> x.getData()).toArray(i -> new double[i][]);
      convolutionController.convolve(inputBuffers, kernelData, outputBuffers);
    } catch (final Throwable e) {
      throw new RuntimeException("Error mapCoords image res " + Arrays.toString(inputDims), e);
    }
    assert Arrays.stream(output).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    
    return new NNResult(output) {
  
      @Override
      public void free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
        assert error.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        if (!isFrozen()) {
          final double[][] inputBuffers = batch.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
          final double[][] outputBuffers = error.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
          final Tensor weightGradient = new Tensor(kernelDims);
          convolutionController.gradient(inputBuffers, weightGradient.getData(), outputBuffers);
          buffer.get(ConvolutionLayer.this, kernelData).addInPlace(weightGradient.getData());
        }
        if (input.isAlive()) {
          final Tensor[] inputBufferTensors = IntStream.range(0, getData().length()).mapToObj(dataIndex -> new Tensor(inputDims)).toArray(i -> new Tensor[i]);
          final double[][] inputBuffers = Arrays.stream(inputBufferTensors).map(x -> x.getData()).toArray(i -> new double[i][]);
          final double[][] outputBuffers = error.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
          convolutionController.backprop(inputBuffers, kernelData, outputBuffers);
          assert Arrays.stream(inputBufferTensors).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
          input.accumulate(buffer, new TensorArray(inputBufferTensors));
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("filter", kernel.toJson(resources, dataSerializer));
    JsonElement paddingX = json.get("paddingX");
    if (null != paddingX && paddingX.isJsonPrimitive()) this.setPaddingX((paddingX.getAsInt()));
    JsonElement paddingY = json.get("paddingY");
    if (null != paddingY && paddingY.isJsonPrimitive()) this.setPaddingY((paddingY.getAsInt()));
    return json;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public ConvolutionLayer setWeights(final DoubleSupplier f) {
    kernel.coordStream(true).forEach(c -> {
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
  public ConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    kernel.coordStream(true).forEach(c -> {
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
  public Integer getPaddingX() {
    return paddingX;
  }
  
  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   * @return the padding x
   */
  public ConvolutionLayer setPaddingX(Integer paddingX) {
    this.paddingX = paddingX;
    return this;
  }
  
  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  public Integer getPaddingY() {
    return paddingY;
  }
  
  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   * @return the padding y
   */
  public ConvolutionLayer setPaddingY(Integer paddingY) {
    this.paddingY = paddingY;
    return this;
  }
}
