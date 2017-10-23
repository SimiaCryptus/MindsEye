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
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Mapped synapse layer.
 */
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
  
  /**
   * Instantiates a new Mapped synapse layer.
   *
   * @param json the json
   */
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
  /**
   * The Output dims.
   */
  public final int[] outputDims;
  /**
   * The Input dims.
   */
  public final int[] inputDims;
  /**
   * The Output size.
   */
  protected final int outputSize;
  /**
   * The Input size.
   */
  protected final int inputSize;
  /**
   * The Mapping matrix.
   */
  protected final int[] mappingMatrix;
  private volatile Tensor weights;
  
  /**
   * Instantiates a new Mapped synapse layer.
   */
  public MappedSynapseLayer() {
    super();
    this.outputDims = null;
    this.weights = null;
    this.inputDims = null;
    this.outputSize = 0;
    this.inputSize = 0;
    this.mappingMatrix = null;
  }
  
  /**
   * Instantiates a new Mapped synapse layer.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   */
  public MappedSynapseLayer(final int[] inputDims, final int[] outputDims) {
    super();
    this.inputDims = Arrays.copyOf(inputDims, inputDims.length);
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.inputSize = Tensor.dim(this.inputDims);
    this.outputSize = Tensor.dim(this.outputDims);
    this.mappingMatrix = new int[this.inputSize * this.outputSize];
  }
  
  /**
   * Concat int [ ].
   *
   * @param a the a
   * @param b the b
   * @return the int [ ]
   */
  public static int[] concat(int[] a, int[] b) {
    int[] c = new int[a.length + b.length];
    for (int i = 0; i < a.length; i++) c[i] = a[i];
    for (int i = 0; i < b.length; i++) c[i + a.length] = b[i];
    return c;
  }
  
  /**
   * Gets mapped index.
   *
   * @param inputCoord  the input coord
   * @param outputCoord the output coord
   * @return the mapped index
   */
  protected abstract int getMappedIndex(Coordinate inputCoord, Coordinate outputCoord);
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... input) {
    double[] expandedWeights = getExpandedWeights();
    Tensor[] outputA = IntStream.range(0, input[0].getData().length()).parallel().mapToObj(dataIndex -> {
      final Tensor inputTensor = input[0].getData().get(dataIndex);
      return multiply2(expandedWeights, inputTensor.getData());
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, input[0]);
  }
  
  /**
   * Get expanded weights double [ ].
   *
   * @return the double [ ]
   */
  protected double[] getExpandedWeights() {
    double[] matrix = new double[mappingMatrix.length];
    double[] data = getWeights().getData();
    for (int i = 0; i < matrix.length; i++) {
      int mappedIndex = this.mappingMatrix[i];
      matrix[i] = (mappedIndex >= 0) ? data[mappedIndex] : 0;
    }
    return matrix;
  }
  
  /**
   * As new synapse layer dense synapse layer.
   *
   * @return the dense synapse layer
   */
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
  
  /**
   * Sets weights.
   *
   * @param data the data
   * @return the weights
   */
  public MappedSynapseLayer setWeights(final double[] data) {
    this.getWeights().set(data);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
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
  
  /**
   * Gets weights.
   *
   * @return the weights
   */
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
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public MappedSynapseLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.getWeights().getData(), i -> f.getAsDouble());
    return this;
  }
  
  /**
   * Build weights tensor.
   *
   * @return the tensor
   */
  protected abstract Tensor buildWeights();
  
  private final class Result extends NNResult {
    private final NNResult result;
    
    private Result(final Tensor[] data, final NNResult result) {
      super(data);
      this.result = result;
    }
    
    private Tensor[] backprop(final TensorList delta, final DeltaSet buffer) {
      double[] expandedWeights = getExpandedWeights();
      Tensor[] passbackA = IntStream.range(0, result.getData().length()).parallel().mapToObj(dataIndex -> {
        final double[] deltaData = delta.get(dataIndex).getData();
        final Tensor passback = new Tensor(this.result.getData().get(dataIndex).getDimensions());
        DenseSynapseLayer.multiplyT(expandedWeights, deltaData, passback.getData());
        return passback;
      }).toArray(i -> new Tensor[i]);
      this.result.accumulate(buffer, new TensorArray(passbackA));
      return passbackA;
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final TensorList delta) {
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
    
    private void learn(final TensorList delta, final DeltaSet buffer) {
      final double[] deltaData0 = delta.get(0).getData();
      final double[] inputData0 = this.result.getData().get(0).getData();
      Delta deltaBuffer = buffer.get(MappedSynapseLayer.this, getWeights());
      
      int threads = 4;
      IntStream.range(0, threads).parallel().forEach(thread -> {
        Tensor buffer1 = new Tensor(MappedSynapseLayer.this.getWeights().getDimensions());
        final Tensor buffer2 = new Tensor(inputData0.length, deltaData0.length);
        IntStream.range(0, result.getData().length()).filter(i -> thread == (i % threads)).forEach(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final double[] inputData = this.result.getData().get(dataIndex).getData();
          assert (deltaData0.length == deltaData.length);
          assert (inputData0.length == inputData.length);
          DenseSynapseLayer.crossMultiply(deltaData, inputData, buffer2.getData());
          buffer1.setAll(0.0);
          accumulateCompactedWeights(buffer2.getData(), buffer1.getData());
          deltaBuffer.accumulate(buffer1.getData());
        });
        try {
          buffer1.release();
          buffer2.release();
        } catch (Throwable e) {
          throw new RuntimeException(e);
        }
      });
      
      
    }
    
  }
}
