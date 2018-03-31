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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Sums all input values to produce a single-element output.
 */
@SuppressWarnings("serial")
public class SumReducerLayer extends LayerBase {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumReducerLayer.class);
  
  /**
   * Instantiates a new Sum reducer layer.
   */
  public SumReducerLayer() {
  }
  
  /**
   * Instantiates a new Sum reducer layer.
   *
   * @param id the id
   */
  protected SumReducerLayer(@Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json sum reducer layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sum reducer layer
   */
  public static SumReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumReducerLayer(json);
  }
  
  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    return new Result(TensorArray.wrap(IntStream.range(0, inObj[0].getData().length()).parallel().mapToDouble(dataIndex -> {
      double sum = 0;
      for (@Nonnull final Result element : inObj) {
        @Nullable Tensor tensor = element.getData().get(dataIndex);
        @Nullable final double[] input = tensor.getData();
        for (final double element2 : input) {
          sum += element2;
        }
        tensor.freeRef();
      }
      return sum;
    }).mapToObj(x -> new Tensor(new double[]{x}, new int[]{1})).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
      for (@Nonnull final Result in_l : inObj) {
        if (in_l.isAlive()) {
          @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, in_l.getData().length()).parallel().mapToObj(dataIndex -> {
            final double delta = data.get(dataIndex).get(0);
            @Nonnull final Tensor passback = new Tensor(in_l.getData().getDimensions());
            for (int i = 0; i < Tensor.length(in_l.getData().getDimensions()); i++) {
              passback.set(i, delta);
            }
            return passback;
          }).toArray(i -> new Tensor[i]));
          in_l.accumulate(buffer, tensorArray);
        }
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(x -> x.getData().freeRef());
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
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
