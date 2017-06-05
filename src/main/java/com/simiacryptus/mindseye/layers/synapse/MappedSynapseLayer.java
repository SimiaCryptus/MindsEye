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
import com.simiacryptus.mindseye.layers.DeltaBuffer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.media.MaxSubsampleLayer;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

public abstract class MappedSynapseLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MappedSynapseLayer.class);
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    json.addProperty("outputSize", outputSize);
    json.addProperty("inputSize", inputSize);
    json.add("mappingMatrix", JsonUtil.getJson(mappingMatrix));
    json.add("weights", weights.getJson());
    return json;
  }
  
  protected MappedSynapseLayer(JsonObject json) {
    super(json);
    this.outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    this.inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    this.outputSize = json.get("outputSize").getAsInt();
    this.inputSize = json.get("inputSize").getAsInt();
    this.mappingMatrix = JsonUtil.getIntArray(json.getAsJsonArray("mappingMatrix"));
    this.weights = Tensor.fromJson(json.getAsJsonObject("weights"));
  }
  
  
  private static final long serialVersionUID = 3538627887600182889L;
  public final int[] outputDims;
  public final int[] inputDims;
  protected final int outputSize;
  protected final int inputSize;
  protected final int[] mappingMatrix;
  private volatile Tensor weights;
  
  public MappedSynapseLayer() {
    super();
    this.outputDims = null;
    this.weights = null;
    this.inputDims = null;
    this.outputSize = 0;
    this.inputSize = 0;
    this.mappingMatrix = null;
  }
  
  public MappedSynapseLayer(final int[] inputDims, final int[] outputDims) {
    super();
    this.inputDims = Arrays.copyOf(inputDims, inputDims.length);
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.inputSize = Tensor.dim(this.inputDims);
    this.outputSize = Tensor.dim(this.outputDims);
    this.mappingMatrix = new int[this.inputSize * this.outputSize];
  }
  
  public static int[] concat(int[] a, int[] b) {
    int[] c = new int[a.length + b.length];
    for (int i = 0; i < a.length; i++) c[i] = a[i];
    for (int i = 0; i < b.length; i++) c[i + a.length] = b[i];
    return c;
  }
  
  protected abstract int getMappedIndex(Coordinate inputCoord, Coordinate outputCoord);
  
  @Override
  public NNResult eval(final NNResult... input) {
    double[] expandedWeights = getExpandedWeights();
    Tensor[] outputA = IntStream.range(0, input[0].data.length).parallel().mapToObj(dataIndex -> {
      final Tensor inputTensor = input[0].data[dataIndex];
      return multiply2(expandedWeights, inputTensor.getData());
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, input[0]);
  }
  
  protected double[] getExpandedWeights() {
    double[] matrix = new double[mappingMatrix.length];
    double[] data = getWeights().getData();
    for (int i = 0; i < matrix.length; i++) {
      int mappedIndex = this.mappingMatrix[i];
      matrix[i] = (mappedIndex >= 0) ? data[mappedIndex] : 0;
    }
    return matrix;
  }
  
  public DenseSynapseLayer asNewSynapseLayer() {
    DenseSynapseLayer returnValue = new DenseSynapseLayer(this.inputDims, this.outputDims);
    returnValue.setWeights(this.getExpandedWeights());
    return returnValue;
  }
  
  private void accumulateCompactedWeights(double[] source, double[] data) {
    assert (source.length == this.mappingMatrix.length);
    for (int i = 0; i < source.length; i++) {
      int mappedIndex = this.mappingMatrix[i];
      if (mappedIndex >= 0) {
        data[mappedIndex] += source[i];
      }
    }
  }
  
  private Tensor multiply2(final double[] wdata, final double[] indata) {
    final Tensor output = new Tensor(this.outputDims);
    DenseSynapseLayer.multiply(wdata, indata, output.getData());
    return output;
  }
  
  public MappedSynapseLayer setWeights(final double[] data) {
    this.getWeights().set(data);
    return this;
  }
  
  public MappedSynapseLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    getWeights().coordStream().parallel().forEach(c -> {
      getWeights().set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(getWeights().getData());
  }
  
  public Tensor getWeights() {
    if (null == this.weights) {
      synchronized (this) {
        if (null == this.weights) {
          this.weights = buildWeights();
          Tensor inputPrototype = new Tensor(this.inputDims);
          Tensor outputPrototype = new Tensor(this.outputDims);
          Tensor prototype = new Tensor(this.inputSize, this.outputSize);
          inputPrototype.coordStream().forEach(inputCoord -> {
            outputPrototype.coordStream().forEach(outputCoord -> {
              int mappedIndex = getMappedIndex(inputCoord, outputCoord);
              assert (mappedIndex < weights.dim());
              mappingMatrix[prototype.index(inputCoord.index, outputCoord.index)] = mappedIndex;
            });
          });
        }
      }
    }
    return weights;
  }
  
  public MappedSynapseLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.getWeights().getData(), i -> f.getAsDouble());
    return this;
  }
  
  protected abstract Tensor buildWeights();
  
  private final class Result extends NNResult {
    private final NNResult result;
    
    private Result(final Tensor[] data, final NNResult result) {
      super(data);
      this.result = result;
    }
    
    private Tensor[] backprop(final Tensor[] delta, final DeltaSet buffer) {
      double[] expandedWeights = getExpandedWeights();
      Tensor[] passbackA = IntStream.range(0, result.data.length).parallel().mapToObj(dataIndex -> {
        final double[] deltaData = delta[dataIndex].getData();
        final Tensor passback = new Tensor(this.result.data[dataIndex].getDims());
        DenseSynapseLayer.multiplyT(expandedWeights, deltaData, passback.getData());
        return passback;
      }).toArray(i -> new Tensor[i]);
      this.result.accumulate(buffer, passbackA);
      return passbackA;
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
      if (!isFrozen()) {
        learn(delta, buffer);
      }
      if (this.result.isAlive()) {
        backprop(delta, buffer);
      }
    }
    
    @Override
    public boolean isAlive() {
      return this.result.isAlive() || !isFrozen();
    }
    
    private void learn(final Tensor[] delta, final DeltaSet buffer) {
      final double[] deltaData0 = delta[0].getData();
      final double[] inputData0 = this.result.data[0].getData();
      DeltaBuffer deltaBuffer = buffer.get(MappedSynapseLayer.this, getWeights());
      
      int threads = 4;
      IntStream.range(0, threads).parallel().forEach(thread -> {
        Tensor buffer1 = new Tensor(MappedSynapseLayer.this.getWeights().getDims());
        final Tensor buffer2 = new Tensor(inputData0.length, deltaData0.length);
        IntStream.range(0, result.data.length).filter(i -> thread == (i % threads)).forEach(dataIndex -> {
          final double[] deltaData = delta[dataIndex].getData();
          final double[] inputData = this.result.data[dataIndex].getData();
          assert (deltaData0.length == deltaData.length);
          assert (inputData0.length == inputData.length);
          DenseSynapseLayer.crossMultiply(deltaData, inputData, buffer2.getData());
          buffer1.setAll(0.0);
          accumulateCompactedWeights(buffer2.getData(), buffer1.getData());
          deltaBuffer.accumulate(buffer1.getData());
        });
        try {
          buffer1.finalize();
          buffer2.finalize();
        } catch (Throwable e) {
          throw new RuntimeException(e);
        }
      });
      
      
    }
    
  }
}
