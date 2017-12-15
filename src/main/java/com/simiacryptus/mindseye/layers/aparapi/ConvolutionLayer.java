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

package com.simiacryptus.mindseye.layers.aparapi;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.Util;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Convolution layer.
 */
public class ConvolutionLayer extends NNLayer {
  
  
  /**
   * The Kernel.
   */
  public final Tensor kernel;
  /**
   * The Skip.
   */
  public final Tensor skip;
  /**
   * The Simple.
   */
  public final boolean simple;
  
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param json the json
   */
  protected ConvolutionLayer(JsonObject json) {
    super(json);
    this.kernel = Tensor.fromJson(json.get("filter"));
    this.skip = Tensor.fromJson(json.get("skip"));
    this.simple = json.getAsJsonPrimitive("simple").getAsBoolean();
  }
  
  /**
   * Instantiates a new Convolution layer.
   */
  protected ConvolutionLayer() {
    this(null, null, true);
  }
  
  /**
   * Instantiates a new Convolution layer.
   *
   * @param kernel the kernel
   * @param skip   the skip
   * @param simple the simple
   */
  protected ConvolutionLayer(Tensor kernel, Tensor skip, boolean simple) {
    super();
    this.simple = simple;
    this.skip = skip;
    if (kernel.getDimensions().length != 3) throw new IllegalArgumentException();
    if (kernel.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if (kernel.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.kernel = kernel;
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
    this(new Tensor(width, height, bands), new Tensor(1, 1), simple);
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
    json.add("filter", kernel.toJson());
    json.add("skip", skip.toJson());
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
    Util.add(f, this.kernel.getData());
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
    
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputDims = batch.get(0).getDimensions();
    int[] kernelDims = this.kernel.getDimensions();
    double[] kernelData = ConvolutionLayer.this.kernel.getData();
    ConvolutionController convolutionController = new ConvolutionController(inputDims, kernelDims, simple);
    Tensor[] output = IntStream.range(0, batch.length())
      .mapToObj(dataIndex -> new Tensor(convolutionController.getOutputDims()))
      .toArray(i -> new Tensor[i]);
    try {
      double[][] inputBuffers = batch.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
      double[][] outputBuffers = Arrays.stream(output).map(x -> x.getData()).toArray(i -> new double[i][]);
      convolutionController.convolve(inputBuffers, kernelData, outputBuffers);
    } catch (Throwable e) {
      throw new RuntimeException("Error mapCoords image res " + Arrays.toString(inputDims), e);
    }
    assert Arrays.stream(output).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    
    return new NNResult(output) {
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
        assert error.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        if (!isFrozen()) {
          double[][] inputBuffers = batch.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
          double[][] outputBuffers = error.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
          final Tensor weightGradient = new Tensor(kernelDims);
          convolutionController.gradient(inputBuffers, weightGradient.getData(), outputBuffers);
          buffer.get(ConvolutionLayer.this, kernelData).addInPlace(weightGradient.getData());
        }
        if (input.isAlive()) {
          Tensor[] inputBufferTensors = IntStream.range(0, getData().length()).mapToObj(dataIndex -> new Tensor(inputDims)).toArray(i -> new Tensor[i]);
          double[][] inputBuffers = Arrays.stream(inputBufferTensors).map(x -> x.getData()).toArray(i -> new double[i][]);
          double[][] outputBuffers = error.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
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
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public ConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.kernel.coordStream().forEach(c -> {
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
  public ConvolutionLayer setWeights(final DoubleSupplier f) {
    this.kernel.coordStream().forEach(c -> {
      this.kernel.set(c, f.getAsDouble());
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.kernel.getData());
  }
  
}
