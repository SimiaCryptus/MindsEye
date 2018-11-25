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
import com.simiacryptus.util.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * A dense matrix operator using vector-matrix multiplication. Represents a fully connected key of synapses, where all
 * inputs are connected to all outputs via seperate coefficients.
 */
@SuppressWarnings("serial")
public class ReshapeLayer extends LayerBase {
  private static final Logger log = LoggerFactory.getLogger(ReshapeLayer.class);
  /**
   * The Output dims.
   */
  @Nullable
  public final int[] outputDims;

  /**
   * Instantiates a new Img eval key.
   */
  private ReshapeLayer() {
    outputDims = null;
  }

  /**
   * Instantiates a new Fully connected key.
   *
   * @param outputDims the output dims
   */
  public ReshapeLayer(@Nonnull final int... outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
  }

  /**
   * Instantiates a new Img eval key.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ReshapeLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
  }

  /**
   * From json img eval key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img eval key
   */
  public static ReshapeLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ReshapeLayer(json, rs);
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    TensorList data = inObj[0].getData();
    @Nonnull int[] inputDims = data.getDimensions();
    ReshapedTensorList reshapedTensorList = new ReshapedTensorList(data, outputDims);
    data.freeRef();
    return new Result(reshapedTensorList, (DeltaSet<UUID> buffer, TensorList delta) -> {
      @Nonnull ReshapedTensorList tensorList = new ReshapedTensorList(delta, inputDims);
      inObj[0].accumulate(buffer, tensorList);
    }) {

      @Override
      protected void _free() {
        for (@Nonnull Result result : inObj) {
          result.freeRef();
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };

  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
