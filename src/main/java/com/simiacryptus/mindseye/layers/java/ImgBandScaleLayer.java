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
import com.simiacryptus.util.io.JsonUtil;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;
import java.util.stream.IntStream;

/**
 * Scales the input using per-color-band coefficients
 */
@SuppressWarnings("serial")
public class ImgBandScaleLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandScaleLayer.class);
  private final @Nullable double[] weights;
  
  /**
   * Instantiates a new Img band scale layer.
   */
  protected ImgBandScaleLayer() {
    super();
    weights = null;
  }
  
  /**
   * Instantiates a new Img band scale layer.
   *
   * @param bands the bands
   */
  public ImgBandScaleLayer(final double... bands) {
    super();
    weights = bands;
  }
  
  
  /**
   * Instantiates a new Img band scale layer.
   *
   * @param json the json
   */
  protected ImgBandScaleLayer(final @NotNull JsonObject json) {
    super(json);
    weights = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }
  
  /**
   * From json img band scale layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img band scale layer
   */
  public static ImgBandScaleLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new ImgBandScaleLayer(json);
  }
  
  /**
   * Add weights img band scale layer.
   *
   * @param f the f
   * @return the img band scale layer
   */
  public @NotNull ImgBandScaleLayer addWeights(final @NotNull DoubleSupplier f) {
    Util.add(f, getWeights());
    return this;
  }
  
  @Override
  public @NotNull NNResult eval(final NNResult... inObj) {
    return eval(inObj[0]);
  }
  
  /**
   * Eval nn result.
   *
   * @param input the input
   * @return the nn result
   */
  public @NotNull NNResult eval(final @NotNull NNResult input) {
    final @Nullable double[] weights = getWeights();
    final TensorList inData = input.getData();
    inData.addRef();
    input.addRef();
    return new NNResult(TensorArray.wrap(inData.stream().parallel()
                                               .map(tensor -> {
                                     if (tensor.getDimensions().length != 3) {
                                       throw new IllegalArgumentException(Arrays.toString(tensor.getDimensions()));
                                     }
                                     if (tensor.getDimensions()[2] != weights.length) {
                                       throw new IllegalArgumentException(String.format("%s: %s does not have %s bands",
                                                                                        getName(), Arrays.toString(tensor.getDimensions()), weights.length));
                                     }
                                     return tensor.mapCoords(c -> tensor.get(c) * weights[c.getCoords()[2]]);
                                               }).toArray(i -> new Tensor[i])), (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList delta) -> {
      if (!isFrozen()) {
        final Delta<NNLayer> deltaBuffer = buffer.get(ImgBandScaleLayer.this, weights);
        IntStream.range(0, delta.length()).forEach(index -> {
          int[] dimensions = delta.getDimensions();
          int z = dimensions[2];
          int y = dimensions[1];
          int x = dimensions[0];
          final double[] array = RecycleBin.DOUBLES.obtain(z);
          final @Nullable double[] deltaArray = delta.get(index).getData();
          final @Nullable double[] inputData = inData.get(index).getData();
          for (int i = 0; i < z; i++) {
            for (int j = 0; j < y * x; j++) {
              //array[i] += deltaArray[i + z * j];
              array[i] += deltaArray[i * x * y + j] * inputData[i * x * y + j];
            }
          }
          assert Arrays.stream(array).allMatch(v -> Double.isFinite(v));
          deltaBuffer.addInPlace(array);
          RecycleBin.DOUBLES.recycle(array, array.length);
        });
      }
      if (input.isAlive()) {
        @NotNull TensorArray tensorArray = TensorArray.wrap(delta.stream()
                                                                 .map(t -> t.mapCoords((c) -> t.get(c) * weights[c.getCoords()[2]]))
                                                                 .toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
  
      @Override
      protected void _free() {
        inData.freeRef();
        input.freeRef();
      }
  
  
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final @NotNull JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getWeights()));
    return json;
  }
  
  /**
   * Get wieghts double [ ].
   *
   * @return the double [ ]
   */
  public @Nullable double[] getWeights() {
    if (!Arrays.stream(weights).allMatch(v -> Double.isFinite(v))) {
      throw new IllegalStateException(Arrays.toString(weights));
    }
    return weights;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public @NotNull ImgBandScaleLayer setWeights(final @NotNull IntToDoubleFunction f) {
    final @Nullable double[] bias = getWeights();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    assert Arrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this;
  }
  
  /**
   * Set nn layer.
   *
   * @param ds the ds
   * @return the nn layer
   */
  public @NotNull NNLayer set(final @NotNull double[] ds) {
    final @Nullable double[] bias = getWeights();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    assert Arrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this;
  }
  
  @Override
  public @NotNull List<double[]> state() {
    return Arrays.asList(getWeights());
  }
}
