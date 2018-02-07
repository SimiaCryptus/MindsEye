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
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

/**
 * Rectified Linear Unit. y=(x&lt;0)?0:x
 */
@SuppressWarnings("serial")
public class ReLuActivationLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ReLuActivationLayer.class);
  private final @Nullable Tensor weights;
  
  /**
   * Instantiates a new Re lu activation layer.
   */
  public ReLuActivationLayer() {
    super();
    weights = new Tensor(1);
    weights.set(0, 1.);
    setFrozen(true);
  }
  
  @Override
  protected void _free() {
    weights.freeRef();
    super._free();
  }
  
  /**
   * Instantiates a new Re lu activation layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ReLuActivationLayer(final @NotNull JsonObject json, Map<String, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
  }
  
  /**
   * From json re lu activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the re lu activation layer
   */
  public static ReLuActivationLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new ReLuActivationLayer(json, rs);
  }
  
  /**
   * Add weights re lu activation layer.
   *
   * @param f the f
   * @return the re lu activation layer
   */
  public @NotNull ReLuActivationLayer addWeights(final @NotNull DoubleSupplier f) {
    Util.add(f, weights.getData());
    return this;
  }
  
  @Override
  public @NotNull NNResult eval(final NNResult... inObj) {
    final NNResult input = inObj[0];
    final TensorList indata = input.getData();
    input.addRef();
    indata.addRef();
    final int itemCnt = indata.length();
    final Tensor[] output = IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
      Tensor tensorElement = indata.get(dataIndex);
      final @NotNull Tensor tensor = tensorElement.multiply(weights.get(0));
      tensorElement.freeRef();
      final @Nullable double[] outputData = tensor.getData();
      for (int i = 0; i < outputData.length; i++) {
        if (outputData[i] < 0) {
          outputData[i] = 0;
        }
      }
      return tensor;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(TensorArray.wrap(output), (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList delta) -> {
      if (!isFrozen()) {
        IntStream.range(0, delta.length()).parallel().forEach(dataIndex -> {
          Tensor deltaTensor = delta.get(dataIndex);
          final @Nullable double[] deltaData = deltaTensor.getData();
          Tensor inputTensor = indata.get(dataIndex);
          final @Nullable double[] inputData = inputTensor.getData();
          final @NotNull Tensor weightDelta = new Tensor(weights.getDimensions());
          final @Nullable double[] weightDeltaData = weightDelta.getData();
          for (int i = 0; i < deltaData.length; i++) {
            weightDeltaData[0] += inputData[i] < 0 ? 0 : deltaData[i] * inputData[i];
          }
          buffer.get(ReLuActivationLayer.this, weights.getData()).addInPlace(weightDeltaData);
          deltaTensor.freeRef();
          inputTensor.freeRef();
          weightDelta.freeRef();
        });
      }
      if (input.isAlive()) {
        final double weight = weights.getData()[0];
        @NotNull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).parallel().mapToObj(dataIndex -> {
          Tensor deltaTensor = delta.get(dataIndex);
          final @Nullable double[] deltaData = deltaTensor.getData();
          Tensor inTensor = indata.get(dataIndex);
          final @Nullable double[] inputData = inTensor.getData();
          final @NotNull int[] dims = inTensor.getDimensions();
          final @NotNull Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, inputData[i] < 0 ? 0 : deltaData[i] * weight);
          }
          inTensor.freeRef();
          deltaTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        input.freeRef();
        indata.freeRef();
      }
  
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
  
    };
  }
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, @NotNull DataSerializer dataSerializer) {
    final @NotNull JsonObject json = super.getJsonStub();
    json.add("weights", weights.toJson(resources, dataSerializer));
    return json;
  }
  
  /**
   * Gets mobility.
   *
   * @return the mobility
   */
  protected double getMobility() {
    return 1;
  }
  
  /**
   * Sets weight.
   *
   * @param data the data
   * @return the weight
   */
  public @NotNull ReLuActivationLayer setWeight(final double data) {
    weights.set(0, data);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public @NotNull ReLuActivationLayer setWeights(final @NotNull DoubleSupplier f) {
    Arrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
    return this;
  }
  
  @Override
  public @NotNull List<double[]> state() {
    return Arrays.asList(weights.getData());
  }
  
}
