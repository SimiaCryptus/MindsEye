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
 * Sums all inputs together, element-by-element, assuming they all have the same dimension.
 */
@SuppressWarnings("serial")
public class SumInputsLayer extends NNLayer {
  
  /**
   * Instantiates a new Sum inputs layer.
   */
  public SumInputsLayer() {
  }
  
  /**
   * Instantiates a new Sum inputs layer.
   *
   * @param id the id
   */
  protected SumInputsLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json sum inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sum inputs layer
   */
  public static SumInputsLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new SumInputsLayer(json);
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    return new NNResult(Arrays.stream(inObj).parallel().map(x -> x.getData()).reduce((l, r) -> {
      assert l.length() == r.length() || 1 == l.length() || 1 == r.length();
      return TensorArray.wrap(IntStream.range(0, l.length()).parallel()
                                       .mapToObj(i -> {
                                        final Tensor left = l.get(1 == l.length() ? 0 : i);
                                        final Tensor right = r.get(1 == r.length() ? 0 : i);
                                        if (right.dim() == 1) {
                                          return left.mapParallel(v -> v + right.get(0));
                                        }
                                        else {
                                          return left.reduceParallel(right, (v1, v2) -> v1 + v2);
                                        }
                                      })
                                       .toArray(i -> new Tensor[i]));
    }).get(), (final DeltaSet<NNLayer> buffer, final TensorList data) -> {
      assert data.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
      for (final NNResult input : inObj) {
        if (input.isAlive()) {
          TensorList data1 = data;
          data1.addRef();
          if (1 < data1.length() && input.getData().length() == 1) {
            data1 = TensorArray.wrap(data1.stream().parallel().reduce((a, b) -> a.add(b)).get());
          }
          if (1 < data1.get(0).dim() && input.getData().get(0).dim() == 1) {
            TensorArray data2 = TensorArray.wrap(data1.stream().map(t -> new Tensor(new double[]{t.sum()})).toArray(i -> new Tensor[i]));
            data1.freeRef();
            data1 = data2;
          }
          input.accumulate(buffer, data1);
          data1.freeRef();
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
        for (final NNResult element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
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
