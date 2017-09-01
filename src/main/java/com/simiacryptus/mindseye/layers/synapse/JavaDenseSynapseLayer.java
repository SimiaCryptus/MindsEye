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

package com.simiacryptus.mindseye.layers.synapse;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.data.Coordinate;
import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.data.TensorArray;
import com.simiacryptus.mindseye.data.TensorList;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Java dense synapse layer.
 */
public class JavaDenseSynapseLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("weights", weights.getJson());
    return json;
  }
  
  /**
   * From json java dense synapse layer.
   *
   * @param json the json
   * @return the java dense synapse layer
   */
  public static JavaDenseSynapseLayer fromJson(JsonObject json) {
    return new JavaDenseSynapseLayer(json);
  }
  
  /**
   * Instantiates a new Java dense synapse layer.
   *
   * @param json the json
   */
  protected JavaDenseSynapseLayer(JsonObject json) {
    super(json);
    this.outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    this.weights = Tensor.fromJson(json.getAsJsonObject("weights"));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(JavaDenseSynapseLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = 3538627887600182889L;
  /**
   * The Output dims.
   */
  public final int[] outputDims;
  private final Tensor weights;
  
  /**
   * Instantiates a new Java dense synapse layer.
   */
  protected JavaDenseSynapseLayer() {
    super();
    this.outputDims = null;
    this.weights = null;
  }
  
  /**
   * Instantiates a new Java dense synapse layer.
   *
   * @param inputs     the inputs
   * @param outputDims the output dims
   */
  public JavaDenseSynapseLayer(final int inputs, final int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new Tensor(inputs, Tensor.dim(outputDims));
    int outs = Tensor.dim(outputDims);
    setWeights(() -> {
      double ratio = Math.sqrt(6. / (inputs + outs));
      double fate = Util.R.get().nextDouble();
      double v = (1 - 2 * fate) * ratio;
      return v;
    });
  }
  
  private static Tensor multiply(final double[] deltaData, final double[] inputData) {
    final Tensor weightDelta = new Tensor(inputData.length, deltaData.length);
    crossMultiply(deltaData, inputData, weightDelta.getData());
    return weightDelta;
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
    for (int o = 0; o < out.length; o++) {
      double sum = 0;
      for (int i = 0; i < in.length; i++) {
        sum += in[i] * matrix[o + out.length * i];
      }
      out[o] = sum;
    }
  }
  
  /**
   * Multiply t.
   *
   * @param matrix the matrix
   * @param in     the in
   * @param out    the out
   */
  public static void multiplyT(final double[] matrix, final double[] in, double[] out) {
    for (int o = 0; o < out.length; o++) {
      double sum = 0;
      for (int i = 0; i < in.length; i++) {
        sum += in[i] * matrix[o * in.length + i];
      }
      out[o] = sum;
    }
  }
  
  /**
   * Add weights java dense synapse layer.
   *
   * @param f the f
   * @return the java dense synapse layer
   */
  public JavaDenseSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.getWeights().getData());
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      return multiply2(this.getWeights().getData(), input.getData());
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  /**
   * Gets mobility.
   *
   * @return the mobility
   */
  protected double getMobility() {
    return 1;
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
  public JavaDenseSynapseLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public JavaDenseSynapseLayer setWeights(final ToDoubleFunction<Coordinate> f) {
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
  public JavaDenseSynapseLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    
    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }
    
    private Tensor[] backprop(final TensorList delta, final DeltaSet buffer) {
      Tensor[] passbackA = IntStream.range(0, inObj.getData().length()).parallel().mapToObj(dataIndex -> {
        final double[] deltaData = delta.get(dataIndex).getData();
        final Tensor r = JavaDenseSynapseLayer.this.getWeights();
        final Tensor passback = new Tensor(this.inObj.getData().get(dataIndex).getDimensions());
        multiplyT(r.getData(), deltaData, passback.getData());
        return passback;
      }).toArray(i -> new Tensor[i]);
      this.inObj.accumulate(buffer, new TensorArray(passbackA));
      return passbackA;
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final TensorList delta) {
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
    
    private void learn(final TensorList delta, final DeltaSet buffer) {
      IntStream.range(0, inObj.getData().length()).parallel().forEach(dataIndex -> {
        final double[] deltaData = delta.get(dataIndex).getData();
        final double[] inputData = this.inObj.getData().get(dataIndex).getData();
        final Tensor weightDelta = multiply(deltaData, inputData);
        buffer.get(JavaDenseSynapseLayer.this, JavaDenseSynapseLayer.this.getWeights()).accumulate(weightDelta.getData());
      });
    }
    
  }
  
}
