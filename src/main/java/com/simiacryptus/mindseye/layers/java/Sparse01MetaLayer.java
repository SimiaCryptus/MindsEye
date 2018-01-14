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

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The type Sparse 01 meta layer.
 */
@SuppressWarnings("serial")
public class Sparse01MetaLayer extends NNLayer {
  
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
  
  /**
   * Instantiates a new Sparse 01 meta layer.
   *
   * @param id the id
   */
  protected Sparse01MetaLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json sparse 01 meta layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sparse 01 meta layer
   */
  public static Sparse01MetaLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    final Sparse01MetaLayer obj = new Sparse01MetaLayer(json);
    obj.sparsity = json.get("sparsity").getAsInt();
    return obj;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    final NNResult input = inObj[0];
    final int itemCnt = input.getData().length();
    final Tensor avgActivationArray = input.getData().get(0).mapIndex((v, c) ->
                                                                        IntStream.range(0, itemCnt)
                                                                                 .mapToDouble(dataIndex -> input.getData().get(dataIndex).get(c))
                                                                                 .average().getAsDouble());
    final Tensor divergenceArray = avgActivationArray.mapIndex((avgActivation, c) -> {
      assert Double.isFinite(avgActivation);
      if (avgActivation > 0 && avgActivation < 1) {
        return sparsity * Math.log(sparsity / avgActivation) + (1 - sparsity) * Math.log((1 - sparsity) / (1 - avgActivation));
      }
      else {
        return 0;
      }
    });
    return new NNResult(divergenceArray) {
  
      @Override
      public void free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        if (input.isAlive()) {
          final Tensor delta = data.get(0);
          final Tensor feedback[] = new Tensor[itemCnt];
          Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
          avgActivationArray.mapIndex((rho, inputCoord) -> {
            final double d = delta.get(inputCoord);
            final double log2 = (1 - sparsity) / (1 - rho);
            final double log3 = sparsity / rho;
            final double value = d * (log2 - log3) / itemCnt;
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
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.addProperty("sparsity", sparsity);
    return json;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
