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
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Randomly selects a fraction of the inputs and sets all other elements to zero.
 */
@SuppressWarnings("serial")
public class DropoutNoiseLayer extends NNLayer implements StochasticComponent {
  
  
  /**
   * The constant randomize.
   */
  public static final ThreadLocal<Random> random = new ThreadLocal<Random>() {
    @Override
    protected Random initialValue() {
      return new Random();
    }
  };
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DropoutNoiseLayer.class);
  /**
   * The Seed.
   */
  long seed = DropoutNoiseLayer.random.get().nextLong();
  private double value;
  
  /**
   * Instantiates a new Dropout noise layer.
   */
  public DropoutNoiseLayer() {
    this(0.5);
  }
  
  /**
   * Instantiates a new Dropout noise layer.
   *
   * @param value the value
   */
  public DropoutNoiseLayer(final double value) {
    super();
    setValue(value);
  }
  
  /**
   * Instantiates a new Dropout noise layer.
   *
   * @param json the json
   */
  protected DropoutNoiseLayer(final @NotNull JsonObject json) {
    super(json);
    value = json.get("value").getAsDouble();
  }
  
  /**
   * From json dropout noise layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the dropout noise layer
   */
  public static DropoutNoiseLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new DropoutNoiseLayer(json);
  }
  
  @Override
  public @NotNull NNResult eval(final NNResult... inObj) {
    final NNResult inputResult = inObj[0];
    inputResult.addRef();
    final TensorList inputData = inputResult.getData();
    final int itemCnt = inputData.length();
    final Tensor[] mask = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final @NotNull Random random = new Random(seed);
      final Tensor input = inputData.get(dataIndex);
      final @Nullable Tensor output = input.map(x -> {
        if (seed == -1) return 1;
        return random.nextDouble() < getValue() ? 0 : (1.0 / getValue());
      });
      return output;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(TensorArray.wrap(IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final @Nullable double[] input = inputData.get(dataIndex).getData();
      final @Nullable double[] maskT = mask[dataIndex].getData();
      final @NotNull Tensor output = new Tensor(inputData.get(dataIndex).getDimensions());
      final @Nullable double[] outputData = output.getData();
      for (int i = 0; i < outputData.length; i++) {
        outputData[i] = input[i] * maskT[i];
      }
      return output;
    }).toArray(i -> new Tensor[i])), (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList delta) -> {
      if (inputResult.isAlive()) {
        @NotNull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final @Nullable double[] deltaData = delta.get(dataIndex).getData();
          final @NotNull int[] dims = inputData.get(dataIndex).getDimensions();
          final @Nullable double[] maskData = mask[dataIndex].getData();
          final @NotNull Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, maskData[i] * deltaData[i]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]));
        inputResult.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
  
      @Override
      protected void _free() {
        inputResult.freeRef();
      }
  
      @Override
      public boolean isAlive() {
        return inputResult.isAlive() || !isFrozen();
      }
  
    };
  }
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final @NotNull JsonObject json = super.getJsonStub();
    json.addProperty("value", value);
    return json;
  }
  
  /**
   * Gets value.
   *
   * @return the value
   */
  public double getValue() {
    return value;
  }
  
  /**
   * Sets value.
   *
   * @param value the value
   * @return the value
   */
  public @NotNull DropoutNoiseLayer setValue(final double value) {
    this.value = value;
    return this;
  }
  
  @Override
  public void shuffle() {
    seed = DropoutNoiseLayer.random.get().nextLong();
  }
  
  @Override
  public void clearNoise() {
    seed = -1;
  }
  
  @Override
  public @NotNull List<double[]> state() {
    return Arrays.asList();
  }
  
  
}
