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
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Adds uniform random gaussian noise to all input elements.
 */
@SuppressWarnings("serial")
public class GaussianNoiseLayer extends LayerBase {


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
  private static final Logger log = LoggerFactory.getLogger(GaussianNoiseLayer.class);
  private long seed = GaussianNoiseLayer.random.get().nextLong();
  private double value;

  /**
   * Instantiates a new Gaussian noise layer.
   */
  public GaussianNoiseLayer() {
    super();
    setValue(1.0);
  }

  /**
   * Instantiates a new Gaussian noise layer.
   *
   * @param json the json
   */
  protected GaussianNoiseLayer(@Nonnull final JsonObject json) {
    super(json);
    value = json.get("value").getAsDouble();
  }

  /**
   * From json gaussian noise layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the gaussian noise layer
   */
  public static GaussianNoiseLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new GaussianNoiseLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result in0 = inObj[0];
    final TensorList inputData = in0.getData();
    final int itemCnt = inputData.length();
    in0.addRef();
    inputData.addRef();
    final Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      @Nonnull final Random random = new Random(seed);
      @Nullable final Tensor input = inputData.get(dataIndex);
      @Nullable final Tensor output = input.map(x -> {
        return x + random.nextGaussian() * getValue();
      });
      input.freeRef();
      return output;
    }).toArray(i -> new Tensor[i]);
    return new Result(TensorArray.wrap(outputA), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (in0.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          Tensor tensor = delta.get(dataIndex);
          @Nullable final double[] deltaData = tensor.getData();
          @Nonnull final Tensor passback = new Tensor(inputData.getDimensions());
          for (int i = 0; i < passback.length(); i++) {
            passback.set(i, deltaData[i]);
          }
          tensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        in0.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      protected void _free() {
        inputData.freeRef();
        in0.freeRef();
      }


      @Override
      public boolean isAlive() {
        return in0.isAlive() || !isFrozen();
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
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
  @Nonnull
  public GaussianNoiseLayer setValue(final double value) {
    this.value = value;
    return this;
  }

  /**
   * Shuffle.
   */
  public void shuffle() {
    seed = GaussianNoiseLayer.random.get().nextLong();
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
