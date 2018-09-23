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
 * Multiplies all inputs together, element-by-element.
 */
@SuppressWarnings("serial")
public class ProductInputsLayer extends LayerBase {

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
  protected ProductInputsLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * From json product inputs layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the product inputs layer
   */
  public static ProductInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ProductInputsLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert inObj.length > 1;
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    for (int i = 1; i < inObj.length; i++) {
      final int dim0 = Tensor.length(inObj[0].getData().getDimensions());
      final int dimI = Tensor.length(inObj[i].getData().getDimensions());
      if (dim0 != 1 && dimI != 1 && dim0 != dimI) {
        throw new IllegalArgumentException(Arrays.toString(inObj[0].getData().getDimensions()) + " != " + Arrays.toString(inObj[i].getData().getDimensions()));
      }
    }
    return new Result(Arrays.stream(inObj).parallel().map(x -> {
      TensorList data = x.getData();
      data.addRef();
      return data;
    }).reduce((l, r) -> {
      TensorArray productArray = TensorArray.wrap(IntStream.range(0, Math.max(l.length(), r.length())).parallel()
          .mapToObj(i1 -> {
            @Nullable final Tensor left = l.get(1 == l.length() ? 0 : i1);
            @Nullable final Tensor right = r.get(1 == r.length() ? 0 : i1);
            Tensor product = Tensor.product(left, right);
            left.freeRef();
            right.freeRef();
            return product;
          }).toArray(i -> new Tensor[i]));
      l.freeRef();
      r.freeRef();
      return productArray;
    }).get(), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      for (@Nonnull final Result input : inObj) {
        if (input.isAlive()) {
          @Nonnull TensorList passback = Arrays.stream(inObj).parallel().map(x -> {
            TensorList tensorList = x == input ? delta : x.getData();
            tensorList.addRef();
            return tensorList;
          }).reduce((l, r) -> {
            TensorArray productList = TensorArray.wrap(IntStream.range(0, Math.max(l.length(), r.length())).parallel()
                .mapToObj(j -> {
                  @Nullable final Tensor left = l.get(1 == l.length() ? 0 : j);
                  @Nullable final Tensor right = r.get(1 == r.length() ? 0 : j);
                  Tensor product = Tensor.product(left, right);
                  left.freeRef();
                  right.freeRef();
                  return product;
                }).toArray(j -> new Tensor[j]));
            l.freeRef();
            r.freeRef();
            return productList;
          }).get();
          final TensorList inputData = input.getData();
          if (1 == inputData.length() && 1 < passback.length()) {
            TensorArray newValue = TensorArray.wrap(passback.stream().reduce((a, b) -> {
              @Nullable Tensor c = a.addAndFree(b);
              b.freeRef();
              return c;
            }).get());
            passback.freeRef();
            passback = newValue;
          }
          if (1 == Tensor.length(inputData.getDimensions()) && 1 < Tensor.length(passback.getDimensions())) {
            TensorArray newValue = TensorArray.wrap(passback.stream()
                .map((a) -> {
                  @Nonnull Tensor b = new Tensor(a.sum());
                  a.freeRef();
                  return b;
                }).toArray(i -> new Tensor[i]));
            passback.freeRef();
            passback = newValue;
          }
          input.accumulate(buffer, passback);
        }
      }
    }) {


      @Override
      public boolean isAlive() {
        for (@Nonnull final Result element : inObj)
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
