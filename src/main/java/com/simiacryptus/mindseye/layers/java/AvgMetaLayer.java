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
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * Computes the average value for each element across all elements of an execution batch. The output batch size will
 * always be one.
 */
@SuppressWarnings("serial")
public class AvgMetaLayer extends LayerBase {


  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgMetaLayer.class);
  /**
   * The Last result.
   */
  @Nullable
  public Tensor lastResult;
  private int minBatchCount = 1;

  /**
   * Instantiates a new Avg meta key.
   */
  public AvgMetaLayer() {
  }

  /**
   * Instantiates a new Avg meta key.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected AvgMetaLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    lastResult = Tensor.fromJson(json.get("lastResult"), resources);
    minBatchCount = json.get("minBatchCount").getAsInt();
  }

  /**
   * From json avg meta key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the avg meta key
   */
  public static AvgMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgMetaLayer(json, rs);
  }

  @Override
  protected void _free() {
    if (null != lastResult) lastResult.freeRef();
    super._free();
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result input = inObj[0];
    input.addRef();
    TensorList inputData = input.getData();
    final int itemCnt = inputData.length();
    @Nullable Tensor thisResult;
    boolean passback;
    if (null == lastResult || inputData.length() > minBatchCount) {
      @Nonnull final ToDoubleFunction<Coordinate> f = (c) ->
          IntStream.range(0, itemCnt)
              .mapToDouble(dataIndex -> {
                Tensor tensor = inputData.get(dataIndex);
                double v = tensor.get(c);
                tensor.freeRef();
                return v;
              })
              .sum() / itemCnt;
      Tensor tensor = inputData.get(0);
      thisResult = tensor.mapCoords(f);
      tensor.freeRef();
      passback = true;
      if (null != lastResult) lastResult.freeRef();
      lastResult = thisResult;
      lastResult.addRef();
    } else {
      passback = false;
      thisResult = lastResult;
      thisResult.freeRef();
    }
    return new Result(TensorArray.create(thisResult), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      if (passback && input.isAlive()) {
        @Nullable final Tensor delta = data.get(0);
        @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
        Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
        thisResult.coordStream(true).forEach((inputCoord) -> {
          for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
            feedback[inputItem].add(inputCoord, delta.get(inputCoord) / itemCnt);
          }
        });
        delta.freeRef();
        @Nonnull TensorArray tensorArray = TensorArray.wrap(feedback);
        input.accumulate(buffer, tensorArray);
      }
    }) {


      @Override
      public boolean isAlive() {
        return input.isAlive();
      }

      @Override
      protected void _free() {
        thisResult.freeRef();
        input.freeRef();
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    if (null != lastResult) {
      json.add("lastResult", lastResult.toJson(resources, dataSerializer));
    }
    json.addProperty("minBatchCount", minBatchCount);
    return json;
  }

  /**
   * The Min batch count.
   *
   * @return the min batch count
   */
  public int getMinBatchCount() {
    return minBatchCount;
  }

  /**
   * Sets min batch count.
   *
   * @param minBatchCount the min batch count
   * @return the min batch count
   */
  @Nonnull
  public AvgMetaLayer setMinBatchCount(final int minBatchCount) {
    this.minBatchCount = minBatchCount;
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
