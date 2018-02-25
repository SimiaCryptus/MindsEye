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

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Sum meta layer.
 */
@SuppressWarnings("serial")
public class SumMetaLayer extends LayerBase {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumMetaLayer.class);
  @Nullable
  private Tensor lastResult;
  private int minBatches = 1;
  
  /**
   * Instantiates a new Sum meta layer.
   */
  public SumMetaLayer() {
  }
  
  /**
   * Instantiates a new Sum meta layer.
   *
   * @param json      the id
   * @param resources the resources
   */
  protected SumMetaLayer(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    lastResult = Tensor.fromJson(json.get("lastResult"), resources);
    minBatches = json.get("minBatches").getAsInt();
  }
  
  /**
   * From json sum meta layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sum meta layer
   */
  public static SumMetaLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new SumMetaLayer(json, rs);
  }
  
  @Nullable
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    final Result input = inObj[0];
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final int itemCnt = input.getData().length();
    if (null == lastResult || minBatches < itemCnt) {
      @javax.annotation.Nonnull final ToDoubleFunction<Coordinate> f = (c) ->
        IntStream.range(0, itemCnt)
          .mapToDouble(dataIndex -> input.getData().get(dataIndex).get(c))
          .sum();
      lastResult = input.getData().get(0).mapCoords(f);
    }
    return new Result(TensorArray.wrap(lastResult), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      if (input.isAlive()) {
        @javax.annotation.Nullable final Tensor delta = data.get(0);
        @javax.annotation.Nonnull final Tensor feedback[] = new Tensor[itemCnt];
        Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
        @javax.annotation.Nonnull final ToDoubleFunction<Coordinate> f = (inputCoord) -> {
          for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
            feedback[inputItem].add(inputCoord, delta.get(inputCoord));
          }
          return 0;
        };
        delta.mapCoords(f);
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(feedback);
        input.accumulate(buffer, tensorArray);
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive();
      }
      
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, @javax.annotation.Nonnull DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    if (null != lastResult) {
      json.add("lastResult", lastResult.toJson(resources, dataSerializer));
    }
    json.addProperty("minBatches", minBatches);
    return json;
  }
  
  /**
   * Gets min batches.
   *
   * @return the min batches
   */
  public int getMinBatches() {
    return minBatches;
  }
  
  /**
   * Sets min batches.
   *
   * @param minBatches the min batches
   * @return the min batches
   */
  @javax.annotation.Nonnull
  public SumMetaLayer setMinBatches(final int minBatches) {
    this.minBatches = minBatches;
    return this;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
