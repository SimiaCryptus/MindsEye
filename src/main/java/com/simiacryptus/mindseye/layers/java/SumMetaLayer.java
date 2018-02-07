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
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Sum meta layer.
 */
@SuppressWarnings("serial")
public class SumMetaLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumMetaLayer.class);
  private @Nullable Tensor lastResult;
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
  protected SumMetaLayer(final @NotNull JsonObject json, Map<String, byte[]> resources) {
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
  public static SumMetaLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new SumMetaLayer(json, rs);
  }
  
  @Override
  public @Nullable NNResult eval(final @NotNull NNResult... inObj) {
    final NNResult input = inObj[0];
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final int itemCnt = input.getData().length();
    if (null == lastResult || minBatches < itemCnt) {
      final @NotNull ToDoubleFunction<Coordinate> f = (c) ->
        IntStream.range(0, itemCnt)
                 .mapToDouble(dataIndex -> input.getData().get(dataIndex).get(c))
                 .sum();
      lastResult = input.getData().get(0).mapCoords(f);
    }
    return new NNResult(TensorArray.wrap(lastResult), (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList data) -> {
      if (input.isAlive()) {
        final Tensor delta = data.get(0);
        final @NotNull Tensor feedback[] = new Tensor[itemCnt];
        Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
        final @NotNull ToDoubleFunction<Coordinate> f = (inputCoord) -> {
          for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
            feedback[inputItem].add(inputCoord, delta.get(inputCoord));
          }
          return 0;
        };
        delta.mapCoords(f);
        @NotNull TensorArray tensorArray = TensorArray.wrap(feedback);
        input.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
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
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, @NotNull DataSerializer dataSerializer) {
    final @NotNull JsonObject json = super.getJsonStub();
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
  public @NotNull SumMetaLayer setMinBatches(final int minBatches) {
    this.minBatches = minBatches;
    return this;
  }
  
  @Override
  public @NotNull List<double[]> state() {
    return Arrays.asList();
  }
}
