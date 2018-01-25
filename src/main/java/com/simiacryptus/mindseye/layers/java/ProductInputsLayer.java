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
  public NNResult eval(final NNResult... inObj) {
    assert inObj.length > 1;
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
        Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    for (int i = 1; i < inObj.length; i++) {
      final int dim0 = Tensor.dim(inObj[0].getData().get(0).getDimensions());
      final int dimI = Tensor.dim(inObj[i].getData().get(0).getDimensions());
      if (dim0 != 1 && dimI != 1 && dim0 != dimI) {
        throw new IllegalArgumentException(Arrays.toString(inObj[0].getData().get(0).getDimensions()) + " != " + Arrays.toString(inObj[i].getData().get(0).getDimensions()));
      }
    }
    return new NNResult(Arrays.stream(inObj).parallel().map(x -> x.getData()).reduce((l, r) -> {
      return TensorArray.wrap(IntStream.range(0, Math.max(l.length(), r.length())).parallel()
                                       .mapToObj(i1 -> {
                                        final Tensor left = l.get(1 == l.length() ? 0 : i1);
                                        final Tensor right = r.get(1 == r.length() ? 0 : i1);
                                        return Tensor.product(left, right);
                                      }).toArray(i -> new Tensor[i]));
    }).get(), (final DeltaSet<NNLayer> buffer, final TensorList delta) -> {
      assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
      for (final NNResult input : inObj) {
        if (input.isAlive()) {
          TensorList passback = Arrays.stream(inObj).parallel().map(x -> x == input ? delta : x.getData()).reduce((l, r) -> {
            return TensorArray.wrap(IntStream.range(0, Math.max(l.length(), r.length())).parallel()
                                             .mapToObj(j -> {
                                              final Tensor left = l.get(1 == l.length() ? 0 : j);
                                              final Tensor right = r.get(1 == r.length() ? 0 : j);
                                              return Tensor.product(left, right);
                                            }).toArray(j -> new Tensor[j]));
          }).get();
          final TensorList inputData = input.getData();
          if (1 == inputData.length() && 1 < passback.length()) {
            passback = TensorArray.wrap(passback.stream().reduce((a, b) -> {
              Tensor c = a.add(b);
              a.freeRef();
              b.freeRef();
              return c;
            }).get());
          }
          if (1 == Tensor.dim(inputData.getDimensions()) && 1 < Tensor.dim(passback.getDimensions())) {
            passback = TensorArray.wrap(passback.stream()
                                                .map((a) -> {
                                                  Tensor b = new Tensor(a.sum());
                                                  a.freeRef();
                                                  return b;
                                                }).toArray(i -> new Tensor[i]));
          }
          input.accumulate(buffer, passback);
          passback.freeRef();
        }
      }
   }) {
      
  
      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
  
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      Arrays.stream(inObj).forEach(x -> x.getData().freeRef());
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
