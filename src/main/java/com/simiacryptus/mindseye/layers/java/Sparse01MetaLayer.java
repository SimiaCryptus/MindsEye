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
import java.util.stream.IntStream;

/**
 * The type Sparse 01 meta layer.
 */
@SuppressWarnings("serial")
public class Sparse01MetaLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("sparsity", sparsity);
    return json;
  }
  
  /**
   * From json sparse 01 meta layer.
   *
   * @param json the json
   * @return the sparse 01 meta layer
   */
  public static Sparse01MetaLayer fromJson(JsonObject json) {
    Sparse01MetaLayer obj = new Sparse01MetaLayer(json);
    obj.sparsity = json.get("sparsity").getAsInt();
    return obj;
  }
  
  /**
   * Instantiates a new Sparse 01 meta layer.
   *
   * @param id the id
   */
  protected Sparse01MetaLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(Sparse01MetaLayer.class);
  
  /**
   * The Sparsity.
   */
  double sparsity = 0.05;
  
  /**
   * Instantiates a new Sparse 01 meta layer.
   */
  public Sparse01MetaLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.getData().length();
    Tensor avgActivationArray = input.getData().get(0).mapIndex((v, c) ->
                                                             IntStream.range(0, itemCnt)
                                                               .mapToDouble(dataIndex -> input.getData().get(dataIndex).get(c))
                                                               .average().getAsDouble());
    Tensor divergenceArray = avgActivationArray.mapIndex((avgActivation, c) -> {
      assert (Double.isFinite(avgActivation));
      if (avgActivation > 0 && avgActivation < 1) {
        return sparsity * Math.log(sparsity / avgActivation) + (1 - sparsity) * Math.log((1 - sparsity) / (1 - avgActivation));
      }
      else {
        return 0;
      }
    });
    return new NNResult(divergenceArray) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (input.isAlive()) {
          Tensor delta = data.get(0);
          Tensor feedback[] = new Tensor[itemCnt];
          Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
          avgActivationArray.mapIndex((rho, inputCoord) -> {
            double d = delta.get(inputCoord);
            double log2 = (1 - sparsity) / (1 - rho);
            double log3 = sparsity / rho;
            double value = d * (log2 - log3) / itemCnt;
            if (Double.isFinite(value)) {
              for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
                //double in = input.data[inputItem].get(inputCoord);
                feedback[inputItem].add(inputCoord, value);
              }
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
