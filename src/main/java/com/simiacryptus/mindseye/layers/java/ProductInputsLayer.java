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

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Multiplies all inputs together, element-by-element.
 */
@SuppressWarnings("serial")
public class ProductInputsLayer extends NNLayer {
  
  /**
   * Instantiates a new Product inputs layer.
   */
  public ProductInputsLayer() {
  }
  
  /**
   * Instantiates a new Product inputs layer.
   *
   * @param id the id
   */
  protected ProductInputsLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json product inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the product inputs layer
   */
  public static ProductInputsLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ProductInputsLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    assert inObj.length > 1;
    for (int i = 1; i < inObj.length; i++) {
      final int dim0 = Tensor.dim(inObj[0].getData().get(0).getDimensions());
      final int dimI = Tensor.dim(inObj[i].getData().get(0).getDimensions());
      if (dim0 != 1 && dimI != 1 && dim0 != dimI) {
        throw new IllegalArgumentException(Arrays.toString(inObj[0].getData().get(0).getDimensions()) + " != " + Arrays.toString(inObj[i].getData().get(0).getDimensions()));
      }
    }
    final TensorList result = Arrays.stream(inObj).parallel().map(x -> x.getData()).reduce((l, r) -> {
      return new TensorArray(IntStream.range(0, Math.max(l.length(), r.length())).parallel()
                                      .mapToObj(i1 -> {
                                        final Tensor left = l.get(1 == l.length() ? 0 : i1);
                                        final Tensor right = r.get(1 == r.length() ? 0 : i1);
                                        return Tensor.product(left, right);
                                      }).toArray(i -> new Tensor[i]));
    }).get();
    return new NNResult(result) {
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList delta) {
        assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        for (final NNResult input : inObj) {
          if (input.isAlive()) {
            TensorList passback = Arrays.stream(inObj).parallel().map(x -> x == input ? delta : x.getData()).reduce((l, r) -> {
              return new TensorArray(IntStream.range(0, Math.max(l.length(), r.length())).parallel()
                                              .mapToObj(j -> {
                                                final Tensor left = l.get(1 == l.length() ? 0 : j);
                                                final Tensor right = r.get(1 == r.length() ? 0 : j);
                                                return Tensor.product(left, right);
                                              }).toArray(j -> new Tensor[j]));
            }).get();
            final TensorList inputData = input.getData();
            if (1 == inputData.length() && 1 < passback.length()) {
              passback = new TensorArray(passback.stream().reduce((a, b) -> a.add(b)).get());
            }
            if (1 == Tensor.dim(inputData.getDimensions()) && 1 < Tensor.dim(passback.getDimensions())) {
              passback = new TensorArray(passback.stream()
                                                 .map((a) -> new Tensor(a.sum())).toArray(i -> new Tensor[i]));
            }
            input.accumulate(buffer, passback);
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
  
      @Override
      public void finalize() {
        Arrays.stream(inObj).forEach(NNResult::finalize);
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
