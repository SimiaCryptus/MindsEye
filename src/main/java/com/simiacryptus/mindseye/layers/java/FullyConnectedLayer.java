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
public class FullyConnectedLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedLayer.class);
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
  protected FullyConnectedLayer() {
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
   * Instantiates a new Fully connected layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected FullyConnectedLayer(final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    weights = Tensor.fromJson(json.get("weights"), resources);
  }
  
  /**
   * Cross multiply.
   *
   * @param rows   the rows
   * @param cols   the cols
   * @param matrix the matrix
   */
  public static void crossMultiply(final double[] rows, final double[] cols, final double[] matrix) {
    int i = 0;
    for (final double c : cols) {
      for (final double r : rows) {
        matrix[i++] = r * c;
      }
    }
  }
  
  /**
   * From json fully connected layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the fully connected layer
   */
  public static FullyConnectedLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new FullyConnectedLayer(json, rs);
  }
  
  /**
   * Multiply.
   *
   * @param matrix the matrix
   * @param in     the in
   * @param out    the out
   */
  public static void multiply(final double[] matrix, final double[] in, final double[] out) {
    final DoubleMatrix matrixObj = new DoubleMatrix(out.length, in.length, matrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
  }
  
  /**
   * Multiply t.
   *
   * @param matrix the matrix
   * @param in     the in
   * @param out    the out
   */
  public static void multiplyT(final double[] matrix, final double[] in, final double[] out) {
    final DoubleMatrix matrixObj = FullyConnectedLayer.transpose(new DoubleMatrix(in.length, out.length, matrix));
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
    RecycleBinLong.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
  }

//  public static void multiplyT(final double[] data, final double[] in, double[] out) {
//    DoubleMatrix matrix = new DoubleMatrix(in.length, out.length, data);
//    DoubleMatrix at = new DoubleMatrix(1, in.length, in);
//    at.mmuli(matrix, new DoubleMatrix(out.length, 1, out));
//  }

//  public static void multiplyT(final double[] data, final double[] in, double[] out) {
//    DoubleMatrix matrix = new DoubleMatrix(in.length, out.length, data);
//    DoubleMatrix at = new DoubleMatrix(in.length, 1, in);
//    double[] r = matrix.transpose().mmul(at).data;
////    double[] r = at.transpose().mmul(matrix).transpose().data;
//    for (int o = 0; o < out.length; o++) out[o] = r[o];
//  }
  
  /**
   * Transpose double matrix.
   *
   * @param doubleMatrix the double matrix
   * @return the double matrix
   */
  public static DoubleMatrix transpose(final DoubleMatrix doubleMatrix) {
    final DoubleMatrix result = new DoubleMatrix(doubleMatrix.columns, doubleMatrix.rows, RecycleBinLong.DOUBLES.obtain(doubleMatrix.length));
    for (int i = 0; i < doubleMatrix.rows; ++i) {
      for (int j = 0; j < doubleMatrix.columns; ++j) {
        result.put(j, i, doubleMatrix.get(i, j));
      }
    }
    return result;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    assert Tensor.dim(inObj[0].getData().getDimensions()) == Tensor.dim(this.inputDims) : Arrays.toString(inObj[0].getData().getDimensions()) + " == " + Arrays.toString(this.inputDims);
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
    final Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      return multiply2(getWeights().getData(), input.getData());
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
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
   * Gets transpose.
   *
   * @return the transpose
   */
  public NNLayer getTranspose() {
    throw new RuntimeException("Not Implemented");
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
  public FullyConnectedLayer set(final DoubleSupplier f) {
    Arrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedLayer set(final IntToDoubleFunction f) {
    weights.set(f);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedLayer setByCoord(final ToDoubleFunction<Coordinate> f) {
    weights.coordStream(true).forEach(c -> {
      weights.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  /**
   * Init spacial.
   *
   * @param radius    the radius
   * @param stiffness the stiffness
   * @param peak      the peak
   */
  public void initSpacial(final double radius, final double stiffness, final double peak) {
    setByCoord((final Coordinate in, final Coordinate out) -> {
      final double[] doubleCoords = IntStream.range(0, in.getCoords().length).mapToDouble(d -> {
        final double from = in.getCoords()[d] * 1.0 / FullyConnectedLayer.this.inputDims[d];
        final double to = out.getCoords()[d] * 1.0 / FullyConnectedLayer.this.outputDims[d];
        return from - to;
      }).toArray();
      final double dist = Math.sqrt(Arrays.stream(doubleCoords).map(x -> x * x).sum());
      final double factor = (1 + Math.tanh(stiffness * (radius - dist))) / 2;
      return peak * factor;
    });
  }
  
  private Tensor multiply2(final double[] wdata, final double[] indata) {
    final Tensor output = new Tensor(outputDims);
    FullyConnectedLayer.multiply(wdata, indata, output.getData());
    return output;
  }
  
  /**
   * Sets weights.
   *
   * @param data the data
   * @return the weights
   */
  public FullyConnectedLayer set(final double[] data) {
    weights.set(data);
    return this;
  }
  
  /**
   * Set fully connected layer.
   *
   * @param data the data
   * @return the fully connected layer
   */
  public FullyConnectedLayer set(final Tensor data) {
    weights.set(data);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedLayer setByCoord(final ToDoubleBiFunction<Coordinate, Coordinate> f) {
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
  public FullyConnectedLayer setWeightsLog(final double value) {
    weights.coordStream(false).forEach(c -> {
      weights.set(c, (FastRandom.random() - 0.5) * Math.pow(10, value));
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(getWeights().getData());
  }
  
  private final class Result extends NNResult {
  
    private final NNResult inObj;
  
    private Result(final Tensor[] outputA, final NNResult inObj) {
      super((final DeltaSet<NNLayer> buffer, final TensorList delta) -> {
        assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        if (!isFrozen()) {
          final Delta<NNLayer> deltaBuffer = buffer.get(FullyConnectedLayer.this, getWeights().getData());
          final int threads = 4;
          IntStream.range(0, threads).parallel().mapToObj(x -> x).flatMap(thread -> {
            final Tensor weightDelta = new Tensor(Tensor.dim(inputDims), Tensor.dim(outputDims));
            return IntStream.range(0, inObj.getData().length()).filter(i -> thread == i % threads).mapToObj(dataIndex -> {
              final double[] deltaData = delta.get(dataIndex).getData();
              final double[] inputData = inObj.getData().get(dataIndex).getData();
              FullyConnectedLayer.crossMultiply(deltaData, inputData, weightDelta.getData());
              return weightDelta.getData();
            });
          }).reduce((a, b) -> ArrayUtil.add(a, b)).map(data -> deltaBuffer.addInPlace(data));
        }
        if (inObj.isAlive()) {
          final TensorList tensorList = new TensorArray(IntStream.range(0, inObj.getData().length()).parallel().mapToObj(dataIndex -> {
            final double[] deltaData = delta.get(dataIndex).getData();
            final Tensor r = getWeights();
            final Tensor passback = new Tensor(inObj.getData().get(dataIndex).getDimensions());
            FullyConnectedLayer.multiplyT(r.getData(), deltaData, passback.getData());
            return passback;
          }).toArray(i -> new Tensor[i]));
          inObj.accumulate(buffer, tensorList);
          tensorList.freeRef();
        }
      }, outputA);
      this.inObj = inObj;
    }
  
    @Override
    public void free() {
      inObj.free();
    }
    
    @Override
    public boolean isAlive() {
      return inObj.isAlive() || !isFrozen();
    }
    
  }
  
}
