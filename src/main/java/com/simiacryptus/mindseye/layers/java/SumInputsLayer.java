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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;

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
  protected SumInputsLayer(@Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json sum inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sum inputs layer
   */
  public static SumInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumInputsLayer(json);
  }
  
  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    return new Result(Arrays.stream(inObj).parallel().map(x -> {
      TensorList data = x.getData();
      data.addRef();
      return data;
    }).reduce((l, r) -> {
      assert l.length() == r.length() || 1 == l.length() || 1 == r.length();
      @Nonnull TensorArray sum = TensorArray.wrap(IntStream.range(0, l.length()).parallel()
        .mapToObj(i -> {
          @Nullable final Tensor left = l.get(1 == l.length() ? 0 : i);
          @Nullable final Tensor right = r.get(1 == r.length() ? 0 : i);
          @Nullable Tensor tensor;
          if (right.length() == 1) {
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
    }).get(), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      for (@Nonnull final Result input : inObj) {
        if (input.isAlive()) {
          @Nonnull TensorList projectedDelta = delta;
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
          if (1 < Tensor.length(projectedDelta.getDimensions()) && Tensor.length(input.getData().getDimensions()) == 1) {
            Tensor[] data = projectedDelta.stream().map(t -> new Tensor(new double[]{t.sum()})).toArray(i -> new Tensor[i]);
            @Nonnull TensorArray data2 = TensorArray.wrap(data);
            projectedDelta.freeRef();
            projectedDelta = data2;
          }
          input.accumulate(buffer, projectedDelta);
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
        for (@Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
      
    };
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
