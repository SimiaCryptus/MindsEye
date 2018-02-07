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
import java.util.stream.IntStream;

/**
 * This activation layer uses a parameterized hyperbolic function. This function, ion various parameterizations, can
 * resemble: x^2, abs(x), x^3, x However, at high +/- x, the behavior is nearly linear.
 */
@SuppressWarnings("serial")
public class HyperbolicActivationLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(HyperbolicActivationLayer.class);
  private final @Nullable Tensor weights;
  private int negativeMode = 1;
  
  /**
   * Instantiates a new Hyperbolic activation layer.
   */
  public HyperbolicActivationLayer() {
    super();
    weights = new Tensor(2);
    weights.set(0, 1.);
    weights.set(1, 1.);
  }
  
  /**
   * Instantiates a new Hyperbolic activation layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected HyperbolicActivationLayer(final @NotNull JsonObject json, Map<String, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
    negativeMode = json.getAsJsonPrimitive("negativeMode").getAsInt();
  }
  
  /**
   * From json hyperbolic activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the hyperbolic activation layer
   */
  public static HyperbolicActivationLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new HyperbolicActivationLayer(json, rs);
  }
  
  @Override
  public @NotNull NNResult eval(final NNResult... inObj) {
    final TensorList indata = inObj[0].getData();
    indata.addRef();
    inObj[0].addRef();
    final int itemCnt = indata.length();
    return new NNResult(TensorArray.wrap(IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = indata.get(dataIndex);
      return input.map(v -> {
        final int sign = v < 0 ? negativeMode : 1;
        final double a = Math.max(0, weights.get(v < 0 ? 1 : 0));
        return sign * (Math.sqrt(Math.pow(a * v, 2) + 1) - a) / a;
      });
    }).toArray(i -> new Tensor[i])), (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList delta) -> {
      if (!isFrozen()) {
        IntStream.range(0, delta.length()).forEach(dataIndex -> {
          final @Nullable double[] deltaData = delta.get(dataIndex).getData();
          final @Nullable double[] inputData = indata.get(dataIndex).getData();
          final @NotNull Tensor weightDelta = new Tensor(weights.getDimensions());
          for (int i = 0; i < deltaData.length; i++) {
            final double d = deltaData[i];
            final double x = inputData[i];
            final int sign = x < 0 ? negativeMode : 1;
            final double a = Math.max(0, weights.getData()[x < 0 ? 1 : 0]);
            weightDelta.add(x < 0 ? 1 : 0, -sign * d / (a * a * Math.sqrt(1 + Math.pow(a * x, 2))));
          }
          buffer.get(HyperbolicActivationLayer.this, weights.getData()).addInPlace(weightDelta.getData());
        });
      }
      if (inObj[0].isAlive()) {
        @NotNull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final @Nullable double[] deltaData = delta.get(dataIndex).getData();
          final @NotNull int[] dims = indata.get(dataIndex).getDimensions();
          final @NotNull Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            final double x = indata.get(dataIndex).getData()[i];
            final double d = deltaData[i];
            final int sign = x < 0 ? negativeMode : 1;
            final double a = Math.max(0, weights.getData()[x < 0 ? 1 : 0]);
            passback.set(i, sign * d * a * x / Math.sqrt(1 + a * x * a * x));
          }
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        indata.freeRef();
        inObj[0].freeRef();
      }
  
  
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    };
    
  }
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, @NotNull DataSerializer dataSerializer) {
    final @NotNull JsonObject json = super.getJsonStub();
    json.add("weights", weights.toJson(resources, dataSerializer));
    json.addProperty("negativeMode", negativeMode);
    return json;
  }
  
  /**
   * Gets scale l.
   *
   * @return the scale l
   */
  public double getScaleL() {
    return 1 / weights.get(1);
  }
  
  /**
   * Gets scale r.
   *
   * @return the scale r
   */
  public double getScaleR() {
    return 1 / weights.get(0);
  }
  
  /**
   * Sets mode asymetric.
   *
   * @return the mode asymetric
   */
  public @NotNull HyperbolicActivationLayer setModeAsymetric() {
    negativeMode = 0;
    return this;
  }
  
  /**
   * Sets mode even.
   *
   * @return the mode even
   */
  public @NotNull HyperbolicActivationLayer setModeEven() {
    negativeMode = 1;
    return this;
  }
  
  /**
   * Sets mode odd.
   *
   * @return the mode odd
   */
  public @NotNull HyperbolicActivationLayer setModeOdd() {
    negativeMode = -1;
    return this;
  }
  
  /**
   * Sets scale.
   *
   * @param scale the scale
   * @return the scale
   */
  public @NotNull HyperbolicActivationLayer setScale(final double scale) {
    weights.set(0, 1 / scale);
    weights.set(1, 1 / scale);
    return this;
  }
  
  @Override
  public @NotNull List<double[]> state() {
    return Arrays.asList(weights.getData());
  }
  
}
