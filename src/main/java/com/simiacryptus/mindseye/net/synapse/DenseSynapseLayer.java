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

package com.simiacryptus.mindseye.net.synapse;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.DeltaBuffer;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

public class DenseSynapseLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);
  private static final long serialVersionUID = 3538627887600182889L;
  public final int[] outputDims;
  public final int[] inputDims;
  public final Tensor weights;
  
  protected DenseSynapseLayer() {
    super();
    this.outputDims = null;
    this.weights = null;
    this.inputDims = null;
  }
  
  public DenseSynapseLayer(final int[] inputDims, final int[] outputDims) {
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
  
  public static void crossMultiply(final double[] rows, final double[] cols, double[] matrix) {
    int i = 0;
    for (final double c : cols) {
      for (final double r : rows) {
        matrix[i++] = r * c;
      }
    }
  }
  
  public static void multiply(final double[] matrix, final double[] in, double[] out) {
    DoubleMatrix matrixObj = new DoubleMatrix(out.length, in.length, matrix);
    double[] r = matrixObj.mmul(new DoubleMatrix(in.length, 1, in)).data;
    for (int o = 0; o < out.length; o++) out[o] = r[o];
  }
  
  public static void multiplyT(final double[] data, final double[] in, double[] out) {
    DoubleMatrix matrix = new DoubleMatrix(in.length, out.length, data);
    DoubleMatrix at = new DoubleMatrix(1, in.length, in);
    at.mmuli(matrix, new DoubleMatrix(out.length, 1, out));
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
  
  private Tensor multiply2(final double[] wdata, final double[] indata) {
    final Tensor output = new Tensor(this.outputDims);
    multiply(wdata, indata, output.getData());
    return output;
  }
  
  public DenseSynapseLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }
  
  public DenseSynapseLayer setWeights2(final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    new Tensor(inputDims).coordStream().parallel().forEach(in -> {
      new Tensor(outputDims).coordStream().parallel().forEach(out -> {
        weights.set(new int[]{in.index, out.index}, f.applyAsDouble(in, out));
      });
    });
    return this;
  }
  
  public DenseSynapseLayer setWeights(final ToDoubleFunction<Coordinate> f) {
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
  
  public DenseSynapseLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }
  
  public void initSpacial(double radius, double stiffness, double peak) {
    setWeights2((Coordinate in, Coordinate out) -> {
      double[] doubleCoords = IntStream.range(0, in.coords.length).mapToDouble(d -> {
        double from = in.coords[d] * 1.0 / DenseSynapseLayer.this.inputDims[d];
        double to = out.coords[d] * 1.0 / DenseSynapseLayer.this.outputDims[d];
        return from - to;
      }).toArray();
      double dist = Math.sqrt(Arrays.stream(doubleCoords).map(x -> x * x).sum());
      double factor = (1 + Math.tanh(stiffness * (radius - dist))) / 2;
      return peak * factor;
    });
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    
    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }
    
    private void backprop(final Tensor[] delta, final DeltaSet buffer) {
      Tensor[] passbackA = IntStream.range(0, inObj.data.length).parallel().mapToObj(dataIndex -> {
        final double[] deltaData = delta[dataIndex].getData();
        final Tensor r = DenseSynapseLayer.this.getWeights();
        final Tensor passback = new Tensor(this.inObj.data[dataIndex].getDims());
        multiplyT(r.getData(), deltaData, passback.getData());
        return passback;
      }).toArray(i -> new Tensor[i]);
      this.inObj.accumulate(buffer, passbackA);
      Arrays.stream(passbackA).forEach(x -> {
        try {
          x.finalize();
        } catch (Throwable throwable) {
          throw new RuntimeException(throwable);
        }
      });
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
      DeltaBuffer deltaBuffer = buffer.get(DenseSynapseLayer.this, DenseSynapseLayer.this.getWeights());
      
      int threads = 4;
      IntStream.range(0, threads).parallel().forEach(thread -> {
        final Tensor weightDelta = new Tensor(Tensor.dim(inputDims), Tensor.dim(outputDims));
        IntStream.range(0, inObj.data.length).filter(i -> thread == (i % threads)).forEach(dataIndex -> {
          final double[] deltaData = delta[dataIndex].getData();
          final double[] inputData = this.inObj.data[dataIndex].getData();
          crossMultiply(deltaData, inputData, weightDelta.getData());
          deltaBuffer.accumulate(weightDelta.getData());
        });
        try {
          weightDelta.finalize();
        } catch (Throwable throwable) {
          throw new RuntimeException(throwable);
        }
      });
    }
    
  }
  
}
