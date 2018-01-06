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
import com.simiacryptus.util.ArrayUtil;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import org.jblas.DoubleMatrix;
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
public class FullyConnectedLayer extends NNLayer implements LayerPrecision<FullyConnectedLayer> {
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedLayer.class);
  /**
   * The Input dims.
   */
  public final int[] inputDims;
  /**
   * The Output dims.
   */
  public final int[] outputDims;
  private final Tensor weights;
  
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   */
  private FullyConnectedLayer() {
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
  public FullyConnectedLayer(final int[] inputDims, final int[] outputDims) {
    final int inputs = Tensor.dim(inputDims);
    this.inputDims = Arrays.copyOf(inputDims, inputDims.length);
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    final int outs = Tensor.dim(outputDims);
    weights = new Tensor(inputs, outs);
    set(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outs + 1));
      final double fate = Util.R.get().nextDouble();
      final double v = (1 - 2 * fate) * ratio;
      return v;
    });
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected FullyConnectedLayer(final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    weights = Tensor.fromJson(json.get("weights"), rs);
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static FullyConnectedLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new FullyConnectedLayer(json, rs);
  }
  
  public static void multiplyT(final double[] matrix, final double[] in, final double[] out) {
    final DoubleMatrix matrixObj = transpose(new DoubleMatrix(in.length, out.length, matrix));
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
    RecycleBin.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
  }
  
  public static void crossMultiply(final double[] rows, final double[] cols, final double[] matrix) {
    int i = 0;
    for (final double c : cols) {
      for (final double r : rows) {
        matrix[i++] = r * c;
      }
    }
  }
  
  public static DoubleMatrix transpose(final DoubleMatrix doubleMatrix) {
    final DoubleMatrix result = new DoubleMatrix(doubleMatrix.columns, doubleMatrix.rows, RecycleBin.DOUBLES.obtain(doubleMatrix.length));
    for (int i = 0; i < doubleMatrix.rows; ++i) {
      for (int j = 0; j < doubleMatrix.columns; ++j) {
        result.put(j, i, doubleMatrix.get(i, j));
      }
    }
    return result;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedLayer set(final DoubleSupplier f) {
    Arrays.parallelSetAll(getWeights().getData(), i -> f.getAsDouble());
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedLayer set(final IntToDoubleFunction f) {
    getWeights().set(f);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedLayer setByCoord(final ToDoubleFunction<Coordinate> f) {
    getWeights().coordStream(true).parallel().forEach(c -> {
      getWeights().set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param data the data
   * @return the weights
   */
  public FullyConnectedLayer set(final double[] data) {
    getWeights().set(data);
    return this;
  }
  
  /**
   * Set fully connected layer.
   *
   * @param data the data
   * @return the fully connected layer
   */
  public FullyConnectedLayer set(final Tensor data) {
    getWeights().set(data);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedLayer setByCoord(final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    new Tensor(inputDims).coordStream(true).parallel().forEach(in -> {
      new Tensor(outputDims).coordStream(true).parallel().forEach(out -> {
        getWeights().set(new int[]{in.getIndex(), out.getIndex()}, f.applyAsDouble(in, out));
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
  public FullyConnectedLayer setWeightsLog(final double value) {
    getWeights().coordStream(true).parallel().forEach(c -> {
      getWeights().set(c, (FastRandom.random() - 0.5) * Math.pow(10, value));
    });
    return this;
  }
  
  /**
   * Gets compatibility layer.
   *
   * @return the compatibility layer
   */
  public NNLayer getCompatibilityLayer() {
    return new com.simiacryptus.mindseye.layers.java.FullyConnectedLayer(inputDims, outputDims).set(getWeights());
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    if (((CudaExecutionContext) nncontext).getDeviceNumber() < 0) return getCompatibilityLayer().eval(nncontext, inObj);
  
    assert Tensor.dim(inObj[0].getData().getDimensions()) == Tensor.dim(this.inputDims) : Arrays.toString(inObj[0].getData().getDimensions()) + " == " + Arrays.toString(this.inputDims);
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
    final Tensor[] output = IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
      return multiply(getWeights().getData(), inObj[0].getData().get(dataIndex).getData());
    }).toArray(i -> new Tensor[i]);
    return new NNResult(output) {
    
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList delta) {
        assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        if (!isFrozen()) {
          learn(delta, buffer);
        }
        if (inObj[0].isAlive()) {
          backprop(delta, buffer);
        }
      }
    
      private void backprop(final TensorList delta, final DeltaSet<NNLayer> buffer) {
        final TensorArray tensorArray = new TensorArray(IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final Tensor r = getWeights();
          final Tensor passback = new Tensor(inObj[0].getData().get(dataIndex).getDimensions());
          multiplyT(r.getData(), deltaData, passback.getData());
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
        tensorArray.recycle();
      }
    
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    
      private void learn(final TensorList delta, final DeltaSet<NNLayer> buffer) {
        final Delta<NNLayer> deltaBuffer = buffer.get(FullyConnectedLayer.this, getWeights().getData());
        final int threads = 4;
        IntStream.range(0, threads).parallel().mapToObj(x -> x).flatMap(thread -> {
          final Tensor weightDelta = new Tensor(Tensor.dim(inputDims), Tensor.dim(outputDims));
          return IntStream.range(0, inObj[0].getData().length()).filter(i -> thread == i % threads).mapToObj(dataIndex -> {
            final double[] deltaData = delta.get(dataIndex).getData();
            final double[] inputData = inObj[0].getData().get(dataIndex).getData();
            crossMultiply(deltaData, inputData, weightDelta.getData());
            return weightDelta.getData();
          });
        }).reduce((a, b) -> ArrayUtil.add(a, b)).map(data -> deltaBuffer.addInPlace(data));
      }
    
    };
  }
  
  private Tensor multiply(final double[] matrix, final double[] vector) {
    final Tensor output = new Tensor(outputDims);
    final double[] out = output.getData();
    multiply(matrix, vector, out);
    return output;
  }
  
  private void multiply(double[] matrix, double[] vector, double[] out) {
    final DoubleMatrix matrixObj = new DoubleMatrix(out.length, vector.length, matrix);
    matrixObj.mmuli(new DoubleMatrix(vector.length, 1, vector), new DoubleMatrix(out.length, 1, out));
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    json.add("weights", getWeights().toJson(resources, dataSerializer));
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(getWeights().getData());
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Override
  public FullyConnectedLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  /**
   * The Weights.
   */
  public Tensor getWeights() {
    return weights;
  }
}
