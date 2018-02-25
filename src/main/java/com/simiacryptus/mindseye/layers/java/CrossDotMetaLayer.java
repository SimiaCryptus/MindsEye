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

/**
 * The type Cross dot meta layer.
 */
@SuppressWarnings("serial")
public class CrossDotMetaLayer extends LayerBase {
  
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
  protected CrossDotMetaLayer(@javax.annotation.Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json cross dot meta layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the cross dot meta layer
   */
  public static CrossDotMetaLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new CrossDotMetaLayer(json);
  }
  
  @Nullable
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    final Result input = inObj[0];
    final TensorList indata = input.getData();
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    indata.addRef();
    final int itemCnt = indata.length();
    final int dim = Tensor.length(indata.getDimensions());
    @javax.annotation.Nonnull final Tensor results = new Tensor(dim, dim);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        if (i == j) {
          continue;
        }
        double v = 0;
        for (int k = 0; k < itemCnt; k++) {
          Tensor tensor = indata.get(k);
          @Nullable final double[] kk = tensor.getData();
          v += kk[i] * kk[j];
          tensor.freeRef();
        }
        results.set(new int[]{i, j}, v);
      }
    }
    return new Result(TensorArray.wrap(results), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      if (input.isAlive()) {
        @javax.annotation.Nullable final Tensor deltaTensor = delta.get(0);
        @javax.annotation.Nonnull final Tensor feedback[] = new Tensor[itemCnt];
        Arrays.parallelSetAll(feedback, i -> new Tensor(dim));
        
        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            if (i == j) {
              continue;
            }
            final double v = deltaTensor.get(i, j);
            for (int k = 0; k < itemCnt; k++) {
              Tensor tensor = indata.get(k);
              @Nullable final double[] kk = tensor.getData();
              feedback[k].add(i, v * kk[j]);
              feedback[k].add(j, v * kk[i]);
              tensor.freeRef();
            }
          }
        }
        deltaTensor.freeRef();
        
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(feedback);
        input.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        indata.freeRef();
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
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
