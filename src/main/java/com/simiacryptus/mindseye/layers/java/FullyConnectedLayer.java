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
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Fully connected layer.
 */
public class FullyConnectedLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedLayer.class);
  /**
   * The Output dims.
   */
  public final int[] outputDims;
  /**
   * The Input dims.
   */
  public final int[] inputDims;
  /**
   * The Weights.
   */
  public final Tensor weights;
  
  /**
   * Instantiates a new Fully connected layer.
   *
   * @param json the json
   */
  protected FullyConnectedLayer(JsonObject json) {
    super(json);
    this.outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    this.inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    this.weights = Tensor.fromJson(json.getAsJsonObject("weights"));
  }
  
  /**
   * Instantiates a new Fully connected layer.
   */
  protected FullyConnectedLayer() {
    super();
    this.outputDims = null;
    this.weights = null;
    this.inputDims = null;
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
    int outs = Tensor.dim(outputDims);
    this.weights = new Tensor(inputs, outs);
    setWeights(() -> {
      double ratio = Math.sqrt(6. / (inputs + outs + 1));
      double fate = Util.R.get().nextDouble();
      double v = (1 - 2 * fate) * ratio;
      return v;
    });
  }
  
  /**
   * From json fully connected layer.
   *
   * @param json the json
   * @return the fully connected layer
   */
  public static FullyConnectedLayer fromJson(JsonObject json) {
    return new FullyConnectedLayer(json);
  }
  
  /**
   * Cross multiply.
   *
   * @param rows   the rows
   * @param cols   the cols
   * @param matrix the matrix
   */
  public static void crossMultiply(final double[] rows, final double[] cols, double[] matrix) {
    int i = 0;
    for (final double c : cols) {
      for (final double r : rows) {
        matrix[i++] = r * c;
      }
    }
  }
  
  /**
   * Multiply.
   *
   * @param matrix the matrix
   * @param in     the in
   * @param out    the out
   */
  public static void multiply(final double[] matrix, final double[] in, double[] out) {
    DoubleMatrix matrixObj = new DoubleMatrix(out.length, in.length, matrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
  }
  
  /**
   * Multiply t.
   *
   * @param matrix the matrix
   * @param in     the in
   * @param out    the out
   */
  public static void multiplyT(final double[] matrix, final double[] in, double[] out) {
    DoubleMatrix matrixObj = transpose(new DoubleMatrix(in.length, out.length, matrix));
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
    DoubleArrays.recycle(matrixObj.data);
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
  public static DoubleMatrix transpose(DoubleMatrix doubleMatrix) {
    DoubleMatrix result = new DoubleMatrix(doubleMatrix.columns, doubleMatrix.rows, DoubleArrays.obtain(doubleMatrix.length));
    for (int i = 0; i < doubleMatrix.rows; ++i) {
      for (int j = 0; j < doubleMatrix.columns; ++j) {
        result.put(j, i, doubleMatrix.get(i, j));
      }
    }
    return result;
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    json.add("weights", weights.getJson());
    return json;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
    Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      return multiply2(this.getWeights().getData(), input.getData());
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  private Tensor multiply2(final double[] wdata, final double[] indata) {
    final Tensor output = new Tensor(this.outputDims);
    multiply(wdata, indata, output.getData());
    return output;
  }
  
  /**
   * Sets weights.
   *
   * @param data the data
   * @return the weights
   */
  public FullyConnectedLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }
  
  /**
   * Sets weights log.
   *
   * @param value the value
   * @return the weights log
   */
  public FullyConnectedLayer setWeightsLog(final double value) {
    this.weights.coordStream().parallel().forEach(c -> {
      this.weights.set(c, (FastRandom.random() - 0.5) * Math.pow(10, value));
    });
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedLayer setWeights(final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    new Tensor(inputDims).coordStream().parallel().forEach(in -> {
      new Tensor(outputDims).coordStream().parallel().forEach(out -> {
        weights.set(new int[]{in.index, out.index}, f.applyAsDouble(in, out));
      });
    });
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public FullyConnectedLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    weights.coordStream().parallel().forEach(c -> {
      weights.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.getWeights().getData());
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
  public FullyConnectedLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }
  
  /**
   * Init spacial.
   *
   * @param radius    the radius
   * @param stiffness the stiffness
   * @param peak      the peak
   */
  public void initSpacial(double radius, double stiffness, double peak) {
    setWeights((Coordinate in, Coordinate out) -> {
      double[] doubleCoords = IntStream.range(0, in.coords.length).mapToDouble(d -> {
        double from = in.coords[d] * 1.0 / FullyConnectedLayer.this.inputDims[d];
        double to = out.coords[d] * 1.0 / FullyConnectedLayer.this.outputDims[d];
        return from - to;
      }).toArray();
      double dist = Math.sqrt(Arrays.stream(doubleCoords).map(x -> x * x).sum());
      double factor = (1 + Math.tanh(stiffness * (radius - dist))) / 2;
      return peak * factor;
    });
  }
  
  /**
   * Gets transpose.
   *
   * @return the transpose
   */
  public NNLayer getTranspose() {
    throw new RuntimeException("Not Implemented");
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    
    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }
    
    private void backprop(final TensorList delta, final DeltaSet buffer) {
      Tensor[] passbackA = IntStream.range(0, inObj.getData().length()).parallel().mapToObj(dataIndex -> {
        final double[] deltaData = delta.get(dataIndex).getData();
        final Tensor r = FullyConnectedLayer.this.getWeights();
        final Tensor passback = new Tensor(this.inObj.getData().get(dataIndex).getDimensions());
        multiplyT(r.getData(), deltaData, passback.getData());
        return passback;
      }).toArray(i -> new Tensor[i]);
      this.inObj.accumulate(buffer, new TensorArray(passbackA));
      Arrays.stream(passbackA).forEach(x -> {
        try {
          x.release();
        } catch (Throwable throwable) {
          throw new RuntimeException(throwable);
        }
      });
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final TensorList delta) {
      assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
      if (!isFrozen()) {
        learn(delta, buffer);
      }
      if (this.inObj.isAlive()) {
        backprop(delta, buffer);
      }
    }
    
    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }
    
    private void learn(final TensorList delta, final DeltaSet<NNLayer> buffer) {
      Delta<NNLayer> deltaBuffer = buffer.get(FullyConnectedLayer.this, FullyConnectedLayer.this.getWeights().getData());
      int threads = 4;
      IntStream.range(0, threads).parallel().mapToObj(x->x).flatMap(thread -> {
        final Tensor weightDelta = new Tensor(Tensor.dim(inputDims), Tensor.dim(outputDims));
        try {
          return IntStream.range(0, inObj.getData().length()).filter(i -> thread == (i % threads)).mapToObj(dataIndex -> {
            final double[] deltaData = delta.get(dataIndex).getData();
            final double[] inputData = this.inObj.getData().get(dataIndex).getData();
            crossMultiply(deltaData, inputData, weightDelta.getData());
            return weightDelta.getData();
          });
        } catch (Throwable throwable) {
          throw new RuntimeException(throwable);
        } finally {
          weightDelta.release();
        }
      }).reduce((a,b)-> ArrayUtil.add(a,b)).map(data->deltaBuffer.addInPlace(data));
    }
    
  }
  
}
