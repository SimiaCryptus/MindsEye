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
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A dense matrix operator using vector-matrix multiplication. Represents a fully connected layer of synapses, where all
 * inputs are connected to all outputs via seperate coefficients.
 */
@SuppressWarnings("serial")
public class FullyConnectedLayer extends LayerBase {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedLayer.class);
  /**
   * The Input dims.
   */
  @Nullable
  public final int[] inputDims;
  /**
   * The Output dims.
   */
  @Nullable
  public final int[] outputDims;
  @Nullable
  private final Tensor weights;
  
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
  public FullyConnectedLayer(@Nonnull final int[] inputDims, @Nonnull final int[] outputDims) {
    final int inputs = Tensor.length(inputDims);
    this.inputDims = Arrays.copyOf(inputDims, inputDims.length);
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    final int outs = Tensor.length(outputDims);
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
  protected FullyConnectedLayer(@Nonnull final JsonObject json, Map<String, byte[]> resources) {
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
  public static void crossMultiply(@Nonnull final double[] rows, @Nonnull final double[] cols, final double[] matrix) {
    int i = 0;
    for (final double c : cols) {
      for (final double r : rows) {
        matrix[i++] = r * c;
      }
    }
  }
  
  /**
   * Cross multiply t.
   *
   * @param rows   the rows
   * @param cols   the cols
   * @param matrix the matrix
   */
  public static void crossMultiplyT(@Nonnull final double[] rows, @Nonnull final double[] cols, final double[] matrix) {
    int i = 0;
    for (final double r : rows) {
      for (final double c : cols) {
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
  public static FullyConnectedLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new FullyConnectedLayer(json, rs);
  }
  
  /**
   * Multiply.
   *
   * @param matrix the matrix
   * @param in     the in
   * @param out    the out
   */
  public static void multiply(final double[] matrix, @Nonnull final double[] in, @Nonnull final double[] out) {
    @Nonnull final DoubleMatrix matrixObj = new DoubleMatrix(out.length, in.length, matrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
  }
  
  /**
   * Multiply t.
   *
   * @param matrix the matrix
   * @param in     the in
   * @param out    the out
   */
  public static void multiplyT(final double[] matrix, @Nonnull final double[] in, @Nonnull final double[] out) {
    @Nonnull DoubleMatrix doubleMatrix = new DoubleMatrix(in.length, out.length, matrix);
    @Nonnull final DoubleMatrix matrixObj = FullyConnectedLayer.transpose(doubleMatrix);
    matrixObj.mmuli(new DoubleMatrix(in.length, 1, in), new DoubleMatrix(out.length, 1, out));
    RecycleBin.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
  }
  
  /**
   * Transpose double matrix.
   *
   * @param doubleMatrix the double matrix
   * @return the double matrix
   */
  @Nonnull
  public static DoubleMatrix transpose(@Nonnull final DoubleMatrix doubleMatrix) {
    @Nonnull final DoubleMatrix result = new DoubleMatrix(doubleMatrix.columns, doubleMatrix.rows, RecycleBin.DOUBLES.obtain(doubleMatrix.length));
    for (int i = 0; i < doubleMatrix.rows; ++i) {
      for (int j = 0; j < doubleMatrix.columns; ++j) {
        result.put(j, i, doubleMatrix.get(i, j));
      }
    }
    return result;
  }
  
  @Override
  protected void _free() {
    weights.freeRef();
    super._free();
  }
  
  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final TensorList indata = inObj[0].getData();
    indata.addRef();
    for (@Nonnull Result result : inObj) {
      result.addRef();
    }
    FullyConnectedLayer.this.addRef();
    assert Tensor.length(indata.getDimensions()) == Tensor.length(this.inputDims) : Arrays.toString(indata.getDimensions()) + " == " + Arrays.toString(this.inputDims);
    @Nonnull DoubleMatrix doubleMatrix = new DoubleMatrix(Tensor.length(indata.getDimensions()), Tensor.length(outputDims), this.weights.getData());
    @Nonnull final DoubleMatrix matrixObj = FullyConnectedLayer.transpose(doubleMatrix);
    @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, indata.length()).parallel().mapToObj(dataIndex -> {
      @Nullable final Tensor input = indata.get(dataIndex);
      @Nullable final Tensor output = new Tensor(outputDims);
      matrixObj.mmuli(new DoubleMatrix(input.length(), 1, input.getData()), new DoubleMatrix(output.length(), 1, output.getData()));
      input.freeRef();
      return output;
    }).toArray(i -> new Tensor[i]));
    RecycleBin.DOUBLES.recycle(matrixObj.data, matrixObj.data.length);
    this.weights.addRef();
    return new Result(tensorArray, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (!isFrozen()) {
        final Delta<Layer> deltaBuffer = buffer.get(FullyConnectedLayer.this, this.weights.getData());
        final int threads = 4;
        IntStream.range(0, threads).parallel().mapToObj(x -> x).flatMap(thread -> {
          @Nullable Stream<Tensor> stream = IntStream.range(0, indata.length()).filter(i -> thread == i % threads).mapToObj(dataIndex -> {
            @Nonnull final Tensor weightDelta = new Tensor(Tensor.length(inputDims), Tensor.length(outputDims));
            Tensor deltaTensor = delta.get(dataIndex);
            Tensor inputTensor = indata.get(dataIndex);
            FullyConnectedLayer.crossMultiplyT(deltaTensor.getData(), inputTensor.getData(), weightDelta.getData());
            inputTensor.freeRef();
            deltaTensor.freeRef();
            return weightDelta;
          });
          return stream;
        }).reduce((a, b) -> {
          @Nullable Tensor c = a.addAndFree(b);
          b.freeRef();
          return c;
        }).map(data -> {
          @Nonnull Delta<Layer> layerDelta = deltaBuffer.addInPlace(data.getData());
          data.freeRef();
          return layerDelta;
        });
        deltaBuffer.freeRef();
      }
      if (inObj[0].isAlive()) {
        @Nonnull final TensorList tensorList = TensorArray.wrap(IntStream.range(0, indata.length()).parallel().mapToObj(dataIndex -> {
          Tensor deltaTensor = delta.get(dataIndex);
          @Nonnull final Tensor passback = new Tensor(indata.getDimensions());
          FullyConnectedLayer.multiply(this.weights.getData(), deltaTensor.getData(), passback.getData());
          deltaTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorList);
      }
    }) {
      
      @Override
      protected void _free() {
        indata.freeRef();
        FullyConnectedLayer.this.freeRef();
        for (@Nonnull Result result : inObj) {
          result.freeRef();
        }
        FullyConnectedLayer.this.weights.freeRef();
      }
      
      @Override
      public boolean isAlive() {
        return !isFrozen() || Arrays.stream(inObj).anyMatch(x -> x.isAlive());
      }
      
    };
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    json.add("weights", getWeights().toJson(resources, dataSerializer));
    return json;
  }
  
  /**
   * Gets transpose.
   *
   * @return the transpose
   */
  @Nonnull
  public Layer getTranspose() {
    throw new RuntimeException("Not Implemented");
  }
  
  /**
   * The Weights.
   */
  /**
   * Gets weights.
   *
   * @return the weights
   */
  @Nullable
  public Tensor getWeights() {
    return weights;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @Nonnull
  public FullyConnectedLayer set(@Nonnull final DoubleSupplier f) {
    Arrays.parallelSetAll(getWeights().getData(), i -> f.getAsDouble());
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @Nonnull
  public FullyConnectedLayer set(@Nonnull final IntToDoubleFunction f) {
    getWeights().set(f);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @Nonnull
  public FullyConnectedLayer setByCoord(@Nonnull final ToDoubleFunction<Coordinate> f) {
    getWeights().coordStream(true).forEach(c -> {
      getWeights().set(c, f.applyAsDouble(c));
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
    setByCoord((@Nonnull final Coordinate in, @Nonnull final Coordinate out) -> {
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
  
  /**
   * Sets weights.
   *
   * @param data the data
   * @return the weights
   */
  @Nonnull
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
  @Nonnull
  public FullyConnectedLayer set(@Nonnull final Tensor data) {
    getWeights().set(data);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @Nonnull
  public FullyConnectedLayer setByCoord(@Nonnull final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    new Tensor(inputDims).coordStream(true).forEach(in -> {
      new Tensor(outputDims).coordStream(true).forEach(out -> {
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
  @Nonnull
  public FullyConnectedLayer setWeightsLog(final double value) {
    getWeights().coordStream(false).forEach(c -> {
      getWeights().set(c, (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value));
    });
    return this;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(getWeights().getData());
  }
  
  
}
