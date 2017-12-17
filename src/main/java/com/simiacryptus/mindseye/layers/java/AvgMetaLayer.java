/*
 * Copyright (c) 2017 by Andrew Charneski.
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

import java.util.Arrays;
import java.util.List;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Avg meta layer.
 */
@SuppressWarnings("serial")
public class AvgMetaLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgMetaLayer.class);
  /**
   * The Last result.
   */
  public Tensor lastResult;
  private int minBatchCount = 1;
  
  /**
   * Instantiates a new Avg meta layer.
   */
  public AvgMetaLayer() {
  }
  
  /**
   * Instantiates a new Avg meta layer.
   *
   * @param json the json
   */
  protected AvgMetaLayer(final JsonObject json) {
    super(json);
    lastResult = Tensor.fromJson(json.get("lastResult"));
    minBatchCount = json.get("minBatchCount").getAsInt();
  }
  
  /**
   * From json avg meta layer.
   *
   * @param json the json
   * @return the avg meta layer
   */
  public static AvgMetaLayer fromJson(final JsonObject json) {
    return new AvgMetaLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final NNResult input = inObj[0];
    final int itemCnt = input.getData().length();
    Tensor thisResult;
    boolean passback;
    if (null == lastResult || input.getData().length() > minBatchCount) {
      final ToDoubleFunction<Coordinate> f = (c) ->
        IntStream.range(0, itemCnt)
          .mapToDouble(dataIndex -> input.getData().get(dataIndex).get(c))
          .sum() / itemCnt;
      thisResult = input.getData().get(0).mapCoords(f);
      passback = true;
      lastResult = thisResult;
    }
    else {
      passback = false;
      thisResult = lastResult;
    }
    return new NNResult(thisResult) {
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        if (passback && input.isAlive()) {
          final Tensor delta = data.get(0);
          final Tensor feedback[] = new Tensor[itemCnt];
          Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
          thisResult.coordStream().forEach((inputCoord) -> {
            for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
              feedback[inputItem].add(inputCoord, delta.get(inputCoord) / itemCnt);
            }
          });
          input.accumulate(buffer, new TensorArray(feedback));
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive();
      }
      
    };
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    if (null != lastResult) {
      json.add("lastResult", lastResult.toJson());
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
  public AvgMetaLayer setMinBatchCount(final int minBatchCount) {
    this.minBatchCount = minBatchCount;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
