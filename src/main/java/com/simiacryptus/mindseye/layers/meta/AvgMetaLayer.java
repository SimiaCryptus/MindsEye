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

package com.simiacryptus.mindseye.layers.meta;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Avg meta layer.
 */
@SuppressWarnings("serial")
public class AvgMetaLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("lastResult", null==lastResult?null:lastResult.getJson());
    return json;
  }

  /**
   * From json avg meta layer.
   *
   * @param json the json
   * @return the avg meta layer
   */
  public static AvgMetaLayer fromJson(JsonObject json) {
    return new AvgMetaLayer(json);
  }

  /**
   * Instantiates a new Avg meta layer.
   *
   * @param id the id
   */
  protected AvgMetaLayer(JsonObject id) {
    super(id);
    if(!id.isJsonNull() && id.has("lastResult")) {
      JsonElement lastResult = id.get("lastResult");
      if(null != lastResult && !lastResult.isJsonNull())
        this.lastResult = Tensor.fromJson(lastResult.getAsJsonObject());
    }
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgMetaLayer.class);
  
  /**
   * Instantiates a new Avg meta layer.
   */
  public AvgMetaLayer() {
  }
  
  /**
   * The Last result.
   */
  public Tensor lastResult;
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.getData().length();
    Tensor result = input.getData().get(0).mapParallel((v, c) ->
                                                  IntStream.range(0, itemCnt)
                                                      .mapToDouble(dataIndex -> input.getData().get(dataIndex).get(c))
                                                      .average().orElse(0.0));
    lastResult = result;
    return new NNResult(result) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (input.isAlive()) {
          Tensor delta = data.get(0);
          Tensor feedback[] = new Tensor[itemCnt];
          Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
          ((null==result)?lastResult:result).mapParallel((rho, inputCoord) -> {
            for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
              feedback[inputItem].add(inputCoord, delta.get(inputCoord) / itemCnt);
            }
            return 0;
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
  public List<double[]> state() {
    return Arrays.asList();
  }
}
