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
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * Adds a bias tensor to the input. Expects a single input of the same dimension as the bias tensor.
 */
@SuppressWarnings("serial")
public class BiasLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  /**
   * The Bias.
   */
  @Nullable
  public final double[] bias;

  /**
   * Instantiates a new Bias key.
   */
  protected BiasLayer() {
    super();
    bias = null;
  }

  /**
   * Instantiates a new Bias key.
   *
   * @param dims the dims
   */
  public BiasLayer(final int... dims) {
    bias = new double[Tensor.length(dims)];
  }


  /**
   * Instantiates a new Bias key.
   *
   * @param json the json
   */
  protected BiasLayer(@Nonnull final JsonObject json) {
    super(json);
    bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }

  /**
   * From json bias key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the bias key
   */
  public static BiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BiasLayer(json);
  }

  /**
   * Add double [ ].
   *
   * @param input the input
   * @return the double [ ]
   */
  public double[] add(@Nonnull final double[] input) {
    final double[] array = RecycleBin.DOUBLES.obtain(input.length);
    if (1 == bias.length) {
      for (int i = 0; i < array.length; i++) {
        array[i] = input[i] + bias[0];
      }
    } else {
      for (int i = 0; i < array.length; i++) {
        array[i] = input[i] + bias[i];
      }
    }
    return array;
  }

  /**
   * Add weights bias key.
   *
   * @param f the f
   * @return the bias key
   */
  @Nonnull
  public BiasLayer addWeights(@Nonnull final DoubleSupplier f) {
    Util.add(f, bias);
    return this;
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    TensorList input;
    if (0 == inObj.length) {
      input = TensorArray.create();
    } else {
      input = inObj[0].getData();
    }
    return new Result(TensorArray.wrap(input.stream().parallel()
        .map(r -> {
          @Nonnull Tensor tensor = new Tensor(add(r.getData()), r.getDimensions());
          r.freeRef();
          return tensor;
        }).toArray(i -> new Tensor[i])),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          if (!isFrozen()) {
            final Delta<UUID> deltaBuffer = buffer.get(BiasLayer.this.getId(), bias);
            if (1 == bias.length) {
              delta.stream().parallel().forEach(d -> {
                @Nullable final double[] array = d.getData();
                deltaBuffer.addInPlace(1 == array.length ? array : new double[]{Arrays.stream(array).sum()});
                d.freeRef();
              });
            } else {
              delta.stream().parallel().forEach(d -> {
                deltaBuffer.addInPlace(d.getData());
                d.freeRef();
              });
            }
            deltaBuffer.freeRef();
          }
          if (0 < inObj.length && inObj[0].isAlive()) {
            delta.addRef();
            inObj[0].accumulate(buffer, delta);
          }
        }) {

      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }


      @Override
      public boolean isAlive() {
        return 0 < inObj.length && inObj[0].isAlive() || !isFrozen();
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(bias));
    return json;
  }


  /**
   * Set nn key.
   *
   * @param ds the ds
   * @return the nn key
   */
  @Nonnull
  public Layer set(@Nonnull final double[] ds) {
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    return this;
  }

  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @Nonnull
  public BiasLayer setWeights(@Nonnull final IntToDoubleFunction f) {
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    return this;
  }

  /**
   * Sets weights log.
   *
   * @param value the value
   * @return the weights log
   */
  @Nonnull
  public BiasLayer setWeightsLog(final double value) {
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value);
    }
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(bias);
  }

  /**
   * Set bias key.
   *
   * @param tensor the tensor
   * @return the bias key
   */
  @Nonnull
  public BiasLayer set(@Nonnull Tensor tensor) {
    assert bias.length == tensor.length();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = tensor.get(i);
    }
    return this;
  }
}
