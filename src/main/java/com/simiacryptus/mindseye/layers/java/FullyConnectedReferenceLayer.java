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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.ArrayUtil;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * A dense matrix operator using vector-matrix multiplication. Represents a fully connected layer of synapses, where all
 * inputs are connected to all outputs via seperate coefficients.
 */
@SuppressWarnings("serial")
public class FullyConnectedReferenceLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedReferenceLayer.class);
  /**
   * The Input dims.
   */
  public final int[] inputDims;
  /**
   * The Output dims.
   */
  public final int[] outputDims;
  /**
   * The Weights.
   */
  public final Tensor weights;
  
  /**
   * Instantiates a new Fully connected layer.
   */
  protected FullyConnectedReferenceLayer() {
    super();
    outputDims = null;
    weights = null;
    inputDims = null;
  }
  
  /**
   * Instantiates a new Fully connected layer.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public FullyConnectedReferenceLayer(final int[] inputDims, final int[] outputDims) {
    this.inputDims = Arrays.copyOf(inputDims, inputDims.length);
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    final int inputs = Tensor.dim(inputDims);
    final int outputs = Tensor.dim(outputDims);
    weights = new Tensor(inputs, outputs);
    set(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outputs + 1));
      final double fate = Util.R.get().nextDouble();
      final double v = (1 - 2 * fate) * ratio;
      return v;
    });
  }
  
  /**
   * Instantiates a new Fully connected layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected FullyConnectedReferenceLayer(final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    weights = Tensor.fromJson(json.get("weights"), resources);
  }
  
  /**
   * From json fully connected layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the fully connected layer
   */
  public static FullyConnectedReferenceLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new FullyConnectedReferenceLayer(json, rs);
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    final NNResult inputResult = inObj[0];
    int[] inputDimensions = inputResult.getData().getDimensions();
    assert Tensor.dim(inputDimensions) == Tensor.dim(this.inputDims) : Arrays.toString(inputDimensions) + " == " + Arrays.toString(this.inputDims);
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
  
    Tensor[] data = IntStream.range(0, inputResult.getData().length()).mapToObj(index -> {
      final Tensor input = inputResult.getData().get(index);
      final Tensor output = new Tensor(outputDims);
      weights.coordStream(false).forEach(c -> {
        int[] coords = c.getCoords();
        double prev = output.get(coords[1]);
        double w = weights.get(c);
        double i = input.get(coords[0]);
        double value = prev + w * i;
        output.set(coords[1], value);
      });
      return output;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(data) {
      
      @Override
      protected void _free() {
        inputResult.free();
      }
      
      @Override
      protected void _accumulate(final DeltaSet<NNLayer> buffer, final TensorList delta) {
        assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        if (!isFrozen()) {
          final Delta<NNLayer> deltaBuffer = buffer.get(FullyConnectedReferenceLayer.this, getWeights().getData());
          Tensor[] array = IntStream.range(0, inputResult.getData().length()).mapToObj(i -> {
            final Tensor input = inputResult.getData().get(i);
            final Tensor output = delta.get(i);
            Tensor weights = new Tensor(FullyConnectedReferenceLayer.this.weights.getDimensions());
            weights.coordStream(false).forEach(c -> {
              int[] coords = c.getCoords();
              weights.set(c, input.get(coords[0]) * output.get(coords[1]));
            });
            return weights;
          }).toArray(i -> new Tensor[i]);
          Arrays.stream(array).map(x -> x.getData()).reduce((a, b) -> ArrayUtil.add(a, b)).map(data1 -> deltaBuffer.addInPlace(data1));
        }
        if (inputResult.isAlive()) {
          final TensorList tensorList = new TensorArray(IntStream.range(0, inputResult.getData().length()).mapToObj(i -> {
            final Tensor input = new Tensor(inputDims);
            final Tensor output = delta.get(i);
            weights.coordStream(false).forEach(c -> {
              int[] coords = c.getCoords();
              input.set(coords[0], input.get(coords[0]) + weights.get(c) * output.get(coords[1]));
            });
            return input;
          }).toArray(i -> new Tensor[i]));
          inputResult.accumulate(buffer, tensorList);
          tensorList.freeRef();
        }
      }
      
      @Override
      public boolean isAlive() {
        return inputResult.isAlive() || !isFrozen();
      }
      
    };
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    json.add("weights", weights.toJson(resources, dataSerializer));
    return json;
  }
  
  
  /**
   * Gets weights.
   *
   * @return the weights
   */
  public Tensor getWeights() {
    return weights;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedReferenceLayer set(final DoubleSupplier f) {
    Arrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedReferenceLayer set(final IntToDoubleFunction f) {
    weights.set(f);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedReferenceLayer setByCoord(final ToDoubleFunction<Coordinate> f) {
    weights.coordStream(true).forEach(c -> {
      weights.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param data the data
   * @return the weights
   */
  public FullyConnectedReferenceLayer set(final double[] data) {
    weights.set(data);
    return this;
  }
  
  /**
   * Set fully connected layer.
   *
   * @param data the data
   * @return the fully connected layer
   */
  public FullyConnectedReferenceLayer set(final Tensor data) {
    weights.set(data);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedReferenceLayer setByCoord(final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    new Tensor(inputDims).coordStream(true).forEach(in -> {
      new Tensor(outputDims).coordStream(true).forEach(out -> {
        weights.set(new int[]{in.getIndex(), out.getIndex()}, f.applyAsDouble(in, out));
      });
    });
    return this;
  }
  
  /**
   * Sets weights log.
   *
   * @param value the value
   * @return the weights log
   */
  public FullyConnectedReferenceLayer setWeightsLog(final double value) {
    weights.coordStream(false).forEach(c -> {
      weights.set(c, (FastRandom.random() - 0.5) * Math.pow(10, value));
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(getWeights().getData());
  }
  
}
