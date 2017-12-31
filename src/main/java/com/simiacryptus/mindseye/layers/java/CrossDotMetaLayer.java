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
import java.util.Map;

/**
 * The type Cross dot meta layer.
 */
@SuppressWarnings("serial")
public class CrossDotMetaLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CrossDotMetaLayer.class);
  
  /**
   * Instantiates a new Cross dot meta layer.
   */
  public CrossDotMetaLayer() {
  }
  
  /**
   * Instantiates a new Cross dot meta layer.
   *
   * @param id the id
   */
  protected CrossDotMetaLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json cross dot meta layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the cross dot meta layer
   */
  public static CrossDotMetaLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new CrossDotMetaLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final NNResult input = inObj[0];
    final int itemCnt = input.getData().length();
    final int dim = input.getData().get(0).dim();
    final Tensor results = new Tensor(dim, dim);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        if (i == j) {
          continue;
        }
        double v = 0;
        for (int k = 0; k < itemCnt; k++) {
          final double[] kk = input.getData().get(k).getData();
          v += kk[i] * kk[j];
        }
        results.set(new int[]{i, j}, v);
      }
    }
    return new NNResult(results) {
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        if (input.isAlive()) {
          final Tensor delta = data.get(0);
          final Tensor feedback[] = new Tensor[itemCnt];
          Arrays.parallelSetAll(feedback, i -> new Tensor(dim));
          
          for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
              if (i == j) {
                continue;
              }
              final double v = delta.get(i, j);
              for (int k = 0; k < itemCnt; k++) {
                final double[] kk = input.getData().get(k).getData();
                feedback[k].add(i, v * kk[j]);
                feedback[k].add(j, v * kk[i]);
              }
            }
          }
          
          
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
    return super.getJsonStub();
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
