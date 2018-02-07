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
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Sums all input values to produce a single-element output.
 */
@SuppressWarnings("serial")
public class SumReducerLayer extends NNLayer {
  
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
  protected SumReducerLayer(@javax.annotation.Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json sum reducer layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the sum reducer layer
   */
  public static SumReducerLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new SumReducerLayer(json);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    return new NNResult(TensorArray.wrap(IntStream.range(0, inObj[0].getData().length()).parallel().mapToDouble(dataIndex -> {
      double sum = 0;
      for (@javax.annotation.Nonnull final NNResult element : inObj) {
        final @Nullable double[] input = element.getData().get(dataIndex).getData();
        for (final double element2 : input) {
          sum += element2;
        }
      }
      return sum;
    }).mapToObj(x -> new Tensor(new double[]{x}, new int[]{1})).toArray(i -> new Tensor[i])), (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      for (@javax.annotation.Nonnull final NNResult in_l : inObj) {
        if (in_l.isAlive()) {
          @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, in_l.getData().length()).parallel().mapToObj(dataIndex -> {
            final double delta = data.get(dataIndex).get(0);
            @javax.annotation.Nonnull final Tensor passback = new Tensor(in_l.getData().get(dataIndex).getDimensions());
            for (int i = 0; i < in_l.getData().get(dataIndex).dim(); i++) {
              passback.set(i, delta);
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
        Arrays.stream(inObj).forEach(x -> x.getData().freeRef());
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
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
