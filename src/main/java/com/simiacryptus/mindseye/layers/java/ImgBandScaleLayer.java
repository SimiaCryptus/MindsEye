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
  private final double[] weights;
  
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
  protected ImgBandScaleLayer(final JsonObject json) {
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
  public static ImgBandScaleLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ImgBandScaleLayer(json);
  }
  
  /**
   * Add weights img band scale layer.
   *
   * @param f the f
   * @return the img band scale layer
   */
  public ImgBandScaleLayer addWeights(final DoubleSupplier f) {
    Util.add(f, getWeights());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    return eval(inObj[0]);
  }
  
  /**
   * Eval nn result.
   *
   * @param input the input
   * @return the nn result
   */
  public NNResult eval(final NNResult input) {
    final double[] weights = getWeights();
    assert input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    final Tensor[] outputA = input.getData().stream().parallel()
                                  .map(tensor -> {
                                    if (tensor.getDimensions().length != 3) {
                                      throw new IllegalArgumentException(Arrays.toString(tensor.getDimensions()));
                                    }
                                    if (tensor.getDimensions()[2] != weights.length) {
                                      throw new IllegalArgumentException(String.format("%s: %s does not have %s bands",
                                                                                       getName(), Arrays.toString(tensor.getDimensions()), weights.length));
                                    }
                                    return tensor.mapCoords(c -> tensor.get(c) * weights[c.getCoords()[2]]);
                                  }).toArray(i -> new Tensor[i]);
    assert Arrays.stream(outputA).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    return new NNResult((final DeltaSet<NNLayer> buffer, final TensorList delta) -> {
      assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
      if (!isFrozen()) {
        final Delta<NNLayer> deltaBuffer = buffer.get(ImgBandScaleLayer.this, weights);
        IntStream.range(0, delta.length()).forEach(index -> {
          int[] dimensions = delta.getDimensions();
          int z = dimensions[2];
          int y = dimensions[1];
          int x = dimensions[0];
          final double[] array = RecycleBinLong.DOUBLES.obtain(z);
          final double[] deltaArray = delta.get(index).getData();
          final double[] inputData = input.getData().get(index).getData();
          for (int i = 0; i < z; i++) {
            for (int j = 0; j < y * x; j++) {
              //array[i] += deltaArray[i + z * j];
              array[i] += deltaArray[i * x * y + j] * inputData[i * x * y + j];
            }
          }
          assert Arrays.stream(array).allMatch(v -> Double.isFinite(v));
          deltaBuffer.addInPlace(array);
          RecycleBinLong.DOUBLES.recycle(array, array.length);
        });
      }
      if (input.isAlive()) {
        TensorArray tensorArray = new TensorArray(delta.stream()
                                                       .map(t -> t.mapCoords((c) -> t.get(c) * weights[c.getCoords()[2]]))
                                                       .toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }, outputA) {
  
      @Override
      public void free() {
        input.free();
      }
  
  
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getWeights()));
    return json;
  }
  
  /**
   * Get wieghts double [ ].
   *
   * @return the double [ ]
   */
  public double[] getWeights() {
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
  public ImgBandScaleLayer setWeights(final IntToDoubleFunction f) {
    final double[] bias = getWeights();
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
  public NNLayer set(final double[] ds) {
    final double[] bias = getWeights();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    assert Arrays.stream(bias).allMatch(v -> Double.isFinite(v));
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(getWeights());
  }
}
