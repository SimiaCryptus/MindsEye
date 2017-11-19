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
 * The type Product layer.
 */
public class ProductLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ProductLayer.class);
  
  /**
   * Instantiates a new Product layer.
   *
   * @param id the id
   */
  protected ProductLayer(JsonObject id) {
    super(id);
  }
  
  /**
   * Instantiates a new Product layer.
   */
  public ProductLayer() {
  }
  
  /**
   * From json product layer.
   *
   * @param json the json
   * @return the product layer
   */
  public static ProductLayer fromJson(JsonObject json) {
    return new ProductLayer(json);
  }
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    double[] sum_A = new double[inObj[0].getData().length()];
    Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).mapToObj(dataIndex -> {
      double sum = 1;
      for (final NNResult element : inObj) {
        final double[] input = element.getData().get(dataIndex).getData();
        for (final double element2 : input) {
          sum *= element2;
        }
      }
      sum_A[dataIndex] = sum;
      return new Tensor(new double[]{sum}, 1);
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        for (final NNResult in_l : inObj) {
          if (in_l.isAlive()) {
            Tensor[] passbackA = IntStream.range(0, inObj[0].getData().length()).mapToObj(dataIndex -> {
              final double delta = data.get(dataIndex).get(0);
              final Tensor passback = new Tensor(in_l.getData().get(dataIndex).getDimensions());
              for (int i = 0; i < in_l.getData().get(dataIndex).dim(); i++) {
                passback.set(i, delta * sum_A[dataIndex] / in_l.getData().get(dataIndex).getData()[i]);
              }
              return passback;
            }).toArray(i -> new Tensor[i]);
            in_l.accumulate(buffer, new TensorArray(passbackA));
          }
        }
      }
      
      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
