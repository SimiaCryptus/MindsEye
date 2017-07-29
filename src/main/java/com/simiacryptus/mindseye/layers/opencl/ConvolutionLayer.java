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

package com.simiacryptus.mindseye.layers.opencl;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

public class ConvolutionLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("kernel", kernel.getJson());
    json.add("skip", skip.getJson());
    json.addProperty("simple", simple);
    return json;
  }
  
  public static ConvolutionLayer fromJson(JsonObject json) {
    return new ConvolutionLayer(json);
  }
  protected ConvolutionLayer(JsonObject json) {
    super(json);
    this.kernel = Tensor.fromJson(json.getAsJsonObject("kernel"));
    this.skip = Tensor.fromJson(json.getAsJsonObject("skip"));
    this.simple = json.getAsJsonPrimitive("simple").getAsBoolean();
  }
  
  
  public final Tensor kernel;
  public final Tensor skip;
  public final boolean simple;
  
  protected ConvolutionLayer() {
    this((Tensor)null, (Tensor)null, true);
  }
  
  protected ConvolutionLayer(Tensor kernel, Tensor skip, boolean simple) {
    super();
    this.simple = simple;
    this.skip = skip;
    if(kernel.getDimensions().length != 3) throw new IllegalArgumentException();
    if(kernel.getDimensions()[0] <= 0) throw new IllegalArgumentException();
    if(kernel.getDimensions()[1] <= 0) throw new IllegalArgumentException();
    if(kernel.getDimensions()[2] <= 0) throw new IllegalArgumentException();
    this.kernel = kernel;
  }
  
  public ConvolutionLayer(final int width, int height, final int inputBands, final int outputBands) {
    this(width, height, inputBands * outputBands);
  }
  
  public ConvolutionLayer(final int width, int height, final int bands, boolean simple) {
    this(new Tensor(width,height,bands), new Tensor(new int[]{1,1}), simple);
    assert(!simple || 0 == (width-1) % 2) : "Simple kernels must have odd width";
    assert(!simple || 0 == (height-1) % 2) : "Simple kernels must have odd height";
  }
  
  public ConvolutionLayer(final int width, int height, final int bands) {
    this(width, height, bands, true);
  }
  
  public ConvolutionLayer(final int width, int height, final int inputBands, final int outputBands, boolean simple) {
    this(width, height, inputBands * outputBands, simple);
  }
  
  public ConvolutionLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.kernel.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    
    final NNResult input = inObj[0];
    final TensorList batch = input.data;
    final int[] inputDims = batch.get(0).getDimensions();
    int[] kernelDims = this.kernel.getDimensions();
    ConvolutionController convolutionController = new ConvolutionController(inputDims, kernelDims, simple);
    Tensor[] output = IntStream.range(0, batch.length())
                           .mapToObj(dataIndex -> new Tensor(convolutionController.getOutputDims()))
                           .toArray(i -> new Tensor[i]);
    try {
      double[][] inputBuffers = batch.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
      double[][] outputBuffers = Arrays.stream(output).map(x -> x.getData()).toArray(i -> new double[i][]);
      convolutionController.convolve(inputBuffers, this.kernel.getData(), outputBuffers);
    } catch (Throwable e) {
      throw new RuntimeException("Error with image res " + Arrays.toString(inputDims),e);
    }
    assert Arrays.stream(output).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
  
    return new NNResult(output) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] error) {
        assert Arrays.stream(error).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        if (!isFrozen()) {
          double[][] inputBuffers = batch.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
          double[][] outputBuffers = Arrays.stream(error).map(x -> x.getData()).toArray(i -> new double[i][]);
          final Tensor kernel = ConvolutionLayer.this.kernel;
          final Tensor weightGradient = new Tensor(kernel.getDimensions());
          convolutionController.gradient(inputBuffers, weightGradient.getData(), outputBuffers);
          buffer.get(ConvolutionLayer.this, kernel).accumulate(weightGradient.getData());
        }
        if (input.isAlive()) {
          Tensor[] inputBufferTensors = IntStream.range(0, data.length()).mapToObj(dataIndex -> new Tensor(inputDims)).toArray(i -> new Tensor[i]);
          double[][] inputBuffers = Arrays.stream(inputBufferTensors).map(x -> x.getData()).toArray(i -> new double[i][]);
          double[][] outputBuffers = Arrays.stream(error).map(x -> x.getData()).toArray(i -> new double[i][]);
          convolutionController.backprop(inputBuffers, ConvolutionLayer.this.kernel.getData(), outputBuffers);
          assert Arrays.stream(inputBufferTensors).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
          input.accumulate(buffer, inputBufferTensors);
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  public ConvolutionLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.kernel.coordStream().parallel().forEach(c -> {
      this.kernel.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  public ConvolutionLayer setWeights(final DoubleSupplier f) {
    this.kernel.coordStream().parallel().forEach(c -> {
      this.kernel.set(c, f.getAsDouble());
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.kernel.getData());
  }
  
}
