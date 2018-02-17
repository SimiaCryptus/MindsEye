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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Sums all inputs together, element-by-element, assuming they all have the same dimension.
 */
@SuppressWarnings("serial")
public class SumInputsLayer extends LayerBase {
  
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
  protected SumInputsLayer(@javax.annotation.Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json sum inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sum inputs layer
   */
  public static SumInputsLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new SumInputsLayer(json);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    return new NNResult(Arrays.stream(inObj).parallel().map(x -> {
      TensorList data = x.getData();
      data.addRef();
      return data;
    }).reduce((l, r) -> {
      assert l.length() == r.length() || 1 == l.length() || 1 == r.length();
      @Nonnull TensorArray sum = TensorArray.wrap(IntStream.range(0, l.length()).parallel()
        .mapToObj(i -> {
          @javax.annotation.Nullable final Tensor left = l.get(1 == l.length() ? 0 : i);
          @javax.annotation.Nullable final Tensor right = r.get(1 == r.length() ? 0 : i);
          @javax.annotation.Nullable Tensor tensor;
          if (right.dim() == 1) {
            tensor = left.mapParallel(v -> v + right.get(0));
          }
          else {
            tensor = left.reduceParallel(right, (v1, v2) -> v1 + v2);
          }
          left.freeRef();
          right.freeRef();
          return tensor;
        })
        .toArray(i -> new Tensor[i]));
      l.freeRef();
      r.freeRef();
      return sum;
    }).get(), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      for (@javax.annotation.Nonnull final NNResult input : inObj) {
        if (input.isAlive()) {
          @javax.annotation.Nonnull TensorList projectedDelta = delta;
          if (1 < projectedDelta.length() && input.getData().length() == 1) {
            projectedDelta = TensorArray.wrap(projectedDelta.stream().parallel().reduce((a, b) -> {
              @Nullable Tensor c = a.addAndFree(b);
              b.freeRef();
              return c;
            }).get());
          }
          else {
            projectedDelta.addRef();
          }
          if (1 < Tensor.dim(projectedDelta.getDimensions()) && Tensor.dim(input.getData().getDimensions()) == 1) {
            Tensor[] data = projectedDelta.stream().map(t -> new Tensor(new double[]{t.sum()})).toArray(i -> new Tensor[i]);
            @javax.annotation.Nonnull TensorArray data2 = TensorArray.wrap(data);
            projectedDelta.freeRef();
            projectedDelta = data2;
          }
          input.accumulate(buffer, projectedDelta);
          projectedDelta.freeRef();
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
