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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Max meta layer.
 */
@SuppressWarnings("serial")
public class MaxMetaLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }

  /**
   * From json max meta layer.
   *
   * @param json the json
   * @return the max meta layer
   */
  public static MaxMetaLayer fromJson(JsonObject json) {
    return new MaxMetaLayer(json);
  }

  /**
   * Instantiates a new Max meta layer.
   *
   * @param id the id
   */
  protected MaxMetaLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxMetaLayer.class);
  
  /**
   * Instantiates a new Max meta layer.
   */
  public MaxMetaLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.data.length();
    int vectorSize = input.data.get(0).dim();
    int[] indicies = new int[vectorSize];
    for(int i=0;i<vectorSize;i++) {
      int itemNumber = i;
      indicies[i] = IntStream.range(0, itemCnt)
                             .mapToObj(x -> x).max(Comparator.comparing(dataIndex -> input.data.get(dataIndex).getData()[itemNumber])).get();
    }
    return new NNResult(input.data.get(0).map((v, c) -> {
      return input.data.get(indicies[c.index]).getData()[c.index];
    })) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (input.isAlive()) {
          Tensor delta = data.get(0);
          Tensor feedback[] = new Tensor[itemCnt];
          Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
          input.data.get(0).coordStream().forEach((inputCoord) -> {
            feedback[indicies[inputCoord.index]].add(inputCoord, delta.get(inputCoord));
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
