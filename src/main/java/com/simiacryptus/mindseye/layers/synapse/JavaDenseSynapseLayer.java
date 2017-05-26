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
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

public class JavaDenseSynapseLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(JavaDenseSynapseLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = 3538627887600182889L;
  public final int[] outputDims;
  private final Tensor weights;
  
  protected JavaDenseSynapseLayer() {
    super();
    this.outputDims = null;
    this.weights = null;
  }
  
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
  
  public static void crossMultiply(final double[] rows, final double[] cols, double[] matrix) {
    int i = 0;
    for (final double c : cols) {
      for (final double r : rows) {
        matrix[i++] = r * c;
      }
    }
  }
  
  public static void multiply(final double[] matrix, final double[] in, double[] out) {
    for (int o = 0; o < out.length; o++) {
      double sum = 0;
      for (int i = 0; i < in.length; i++) {
        sum += in[i] * matrix[o + out.length * i];
      }
      out[o] = sum;
    }
  }
  
  public static void multiplyT(final double[] matrix, final double[] in, double[] out) {
    for (int o = 0; o < out.length; o++) {
      double sum = 0;
      for (int i = 0; i < in.length; i++) {
        sum += in[i] * matrix[o * in.length + i];
      }
      out[o] = sum;
    }
  }
  
  public JavaDenseSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.getWeights().getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    Tensor[] outputA = IntStream.range(0, inObj[0].data.length).parallel().mapToObj(dataIndex -> {
      final Tensor input = inObj[0].data[dataIndex];
      return multiply2(this.getWeights().getData(), input.getData());
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("weights", this.getWeights().toString());
    return json;
  }
  
  protected double getMobility() {
    return 1;
  }
  
  private Tensor multiply2(final double[] wdata, final double[] indata) {
    final Tensor output = new Tensor(this.outputDims);
    multiply(wdata, indata, output.getData());
    return output;
  }
  
  public JavaDenseSynapseLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }
  
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
  
  public Tensor getWeights() {
    return weights;
  }
  
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
    
    private Tensor[] backprop(final Tensor[] delta, final DeltaSet buffer) {
      Tensor[] passbackA = IntStream.range(0, inObj.data.length).parallel().mapToObj(dataIndex -> {
        final double[] deltaData = delta[dataIndex].getData();
        final Tensor r = JavaDenseSynapseLayer.this.getWeights();
        final Tensor passback = new Tensor(this.inObj.data[dataIndex].getDims());
        multiplyT(r.getData(), deltaData, passback.getData());
        return passback;
      }).toArray(i -> new Tensor[i]);
      this.inObj.accumulate(buffer, passbackA);
      return passbackA;
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
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
    
    private void learn(final Tensor[] delta, final DeltaSet buffer) {
      IntStream.range(0, inObj.data.length).parallel().forEach(dataIndex -> {
        final double[] deltaData = delta[dataIndex].getData();
        final double[] inputData = this.inObj.data[dataIndex].getData();
        final Tensor weightDelta = multiply(deltaData, inputData);
        buffer.get(JavaDenseSynapseLayer.this, JavaDenseSynapseLayer.this.getWeights()).accumulate(weightDelta.getData());
      });
    }
    
  }
  
}