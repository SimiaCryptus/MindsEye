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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

/**
 * This activation key uses a parameterized hyperbolic function. This function, ion various parameterizations, can
 * resemble: x^2, abs(x), x^3, x However, at high +/- x, the behavior is nearly linear.
 */
@SuppressWarnings("serial")
public class HyperbolicActivationLayer extends LayerBase {


  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(HyperbolicActivationLayer.class);
  @Nullable
  private final Tensor weights;
  private int negativeMode = 1;

  /**
   * Instantiates a new Hyperbolic activation key.
   */
  public HyperbolicActivationLayer() {
    super();
    weights = new Tensor(2);
    weights.set(0, 1.);
    weights.set(1, 1.);
  }

  /**
   * Instantiates a new Hyperbolic activation key.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected HyperbolicActivationLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
    negativeMode = json.getAsJsonPrimitive("negativeMode").getAsInt();
  }

  /**
   * From json hyperbolic activation key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the hyperbolic activation key
   */
  public static HyperbolicActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new HyperbolicActivationLayer(json, rs);
  }

  @Override
  protected void _free() {
    weights.freeRef();
    super._free();
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final TensorList indata = inObj[0].getData();
    indata.addRef();
    inObj[0].addRef();
    weights.addRef();
    HyperbolicActivationLayer.this.addRef();
    final int itemCnt = indata.length();
    return new Result(TensorArray.wrap(IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      @Nullable final Tensor input = indata.get(dataIndex);
      @Nullable Tensor map = input.map(v -> {
        final int sign = v < 0 ? negativeMode : 1;
        final double a = Math.max(0, weights.get(v < 0 ? 1 : 0));
        return sign * (Math.sqrt(Math.pow(a * v, 2) + 1) - a) / a;
      });
      input.freeRef();
      return map;
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      if (!isFrozen()) {
        IntStream.range(0, delta.length()).forEach(dataIndex -> {
          @Nullable Tensor deltaI = delta.get(dataIndex);
          @Nullable Tensor inputI = indata.get(dataIndex);
          @Nullable final double[] deltaData = deltaI.getData();
          @Nullable final double[] inputData = inputI.getData();
          @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
          for (int i = 0; i < deltaData.length; i++) {
            final double d = deltaData[i];
            final double x = inputData[i];
            final int sign = x < 0 ? negativeMode : 1;
            final double a = Math.max(0, weights.getData()[x < 0 ? 1 : 0]);
            weightDelta.add(x < 0 ? 1 : 0, -sign * d / (a * a * Math.sqrt(1 + Math.pow(a * x, 2))));
          }
          deltaI.freeRef();
          inputI.freeRef();
          buffer.get(HyperbolicActivationLayer.this.getId(), weights.getData()).addInPlace(weightDelta.getData()).freeRef();
          weightDelta.freeRef();
        });
      }
      if (inObj[0].isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          @Nullable Tensor inputTensor = indata.get(dataIndex);
          Tensor deltaTensor = delta.get(dataIndex);
          @Nullable final double[] deltaData = deltaTensor.getData();
          @Nonnull final int[] dims = indata.getDimensions();
          @Nonnull final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.length(); i++) {
            final double x = inputTensor.getData()[i];
            final double d = deltaData[i];
            final int sign = x < 0 ? negativeMode : 1;
            final double a = Math.max(0, weights.getData()[x < 0 ? 1 : 0]);
            passback.set(i, sign * d * a * x / Math.sqrt(1 + a * x * a * x));
          }
          deltaTensor.freeRef();
          inputTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      protected void _free() {
        indata.freeRef();
        inObj[0].freeRef();
        weights.freeRef();
        HyperbolicActivationLayer.this.freeRef();
      }


      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    };

  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
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
  @Nonnull
  public HyperbolicActivationLayer setModeAsymetric() {
    negativeMode = 0;
    return this;
  }

  /**
   * Sets mode even.
   *
   * @return the mode even
   */
  @Nonnull
  public HyperbolicActivationLayer setModeEven() {
    negativeMode = 1;
    return this;
  }

  /**
   * Sets mode odd.
   *
   * @return the mode odd
   */
  @Nonnull
  public HyperbolicActivationLayer setModeOdd() {
    negativeMode = -1;
    return this;
  }

  /**
   * Sets scale.
   *
   * @param scale the scale
   * @return the scale
   */
  @Nonnull
  public HyperbolicActivationLayer setScale(final double scale) {
    weights.set(0, 1 / scale);
    weights.set(1, 1 / scale);
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(weights.getData());
  }

}
