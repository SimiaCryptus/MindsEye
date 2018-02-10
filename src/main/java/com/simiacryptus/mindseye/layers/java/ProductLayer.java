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
import java.util.stream.IntStream;

/**
 * The type Product layer.
 */
@SuppressWarnings("serial")
public class ProductLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ProductLayer.class);
  
  /**
   * Instantiates a new Product layer.
   */
  public ProductLayer() {
  }
  
  /**
   * Instantiates a new Product layer.
   *
   * @param id the id
   */
  protected ProductLayer(@javax.annotation.Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json product layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the product layer
   */
  public static ProductLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ProductLayer(json);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    @javax.annotation.Nonnull final double[] sum_A = new double[inObj[0].getData().length()];
    final Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).mapToObj(dataIndex -> {
      double sum = 1;
      for (@javax.annotation.Nonnull final NNResult element : inObj) {
        @Nullable final double[] input = element.getData().get(dataIndex).getData();
        for (final double element2 : input) {
          sum *= element2;
        }
      }
      sum_A[dataIndex] = sum;
      return new Tensor(new double[]{sum}, 1);
    }).toArray(i -> new Tensor[i]);
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    return new NNResult(TensorArray.wrap(outputA), (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      for (@javax.annotation.Nonnull final NNResult in_l : inObj) {
        if (in_l.isAlive()) {
          @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, inObj[0].getData().length()).mapToObj(dataIndex -> {
            final double delta = data.get(dataIndex).get(0);
            @javax.annotation.Nonnull final Tensor passback = new Tensor(in_l.getData().get(dataIndex).getDimensions());
            for (int i = 0; i < in_l.getData().get(dataIndex).dim(); i++) {
              passback.set(i, delta * sum_A[dataIndex] / in_l.getData().get(dataIndex).getData()[i]);
            }
            return passback;
          }).toArray(i -> new Tensor[i]));
          in_l.accumulate(buffer, tensorArray);
          tensorArray.freeRef();
        }
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
        Arrays.stream(inObj).forEach(x -> x.getData().freeRef());
      }
      
      
      @Override
      public boolean isAlive() {
        for (@javax.annotation.Nonnull final NNResult element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
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
