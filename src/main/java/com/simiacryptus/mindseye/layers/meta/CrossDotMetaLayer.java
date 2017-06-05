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
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.reducers.SumReducerLayer;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

@SuppressWarnings("serial")
public class CrossDotMetaLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static CrossDotMetaLayer fromJson(JsonObject json) {
    return new CrossDotMetaLayer(json);
  }
  protected CrossDotMetaLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CrossDotMetaLayer.class);
  
  public CrossDotMetaLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.data.length;
    int dim = input.data[0].dim();
    Tensor results = new Tensor(dim, dim);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        if (i == j) continue;
        double v = 0;
        for (int k = 0; k < itemCnt; k++) {
          double[] kk = input.data[k].getData();
          v += kk[i] * kk[j];
        }
        results.set(new int[]{i, j}, v);
      }
    }
    return new NNResult(results) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (input.isAlive()) {
          Tensor delta = data[0];
          Tensor feedback[] = new Tensor[itemCnt];
          Arrays.parallelSetAll(feedback, i -> new Tensor(dim));
          
          for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
              if (i == j) continue;
              double v = delta.get(i, j);
              for (int k = 0; k < itemCnt; k++) {
                double[] kk = input.data[k].getData();
                feedback[k].add(i, v * kk[j]);
                feedback[k].add(j, v * kk[i]);
              }
            }
          }
          
          
          input.accumulate(buffer, feedback);
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
