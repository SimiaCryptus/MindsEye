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
 * Computes the average value across all elements of each input tensor. The output dimensions are always 1x1x1.
 */
@SuppressWarnings("serial")
public class AvgReducerLayer extends LayerBase {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumReducerLayer.class);
  
  /**
   * Instantiates a new Avg reducer layer.
   */
  public AvgReducerLayer() {
  }
  
  /**
   * Instantiates a new Avg reducer layer.
   *
   * @param id the id
   */
  protected AvgReducerLayer(@javax.annotation.Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json avg reducer layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the avg reducer layer
   */
  public static AvgReducerLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new AvgReducerLayer(json);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    Arrays.stream(inObj).forEach(x -> x.addRef());
    Arrays.stream(inObj).forEach(x -> x.getData().addRef());
    return new NNResult(TensorArray.wrap(IntStream.range(0, inObj[0].getData().length()).parallel().mapToDouble(dataIndex -> {
      double sum = 0;
      for (@javax.annotation.Nonnull final NNResult element : inObj) {
        Tensor tensor = element.getData().get(dataIndex);
        @Nullable final double[] input = tensor.getData();
        for (final double element2 : input) {
          sum += element2 / input.length;
        }
        tensor.freeRef();
      }
      return sum;
    }).mapToObj(x -> new Tensor(new double[]{x}, new int[]{1})).toArray(i -> new Tensor[i])), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      for (@javax.annotation.Nonnull final NNResult in_l : inObj) {
        if (in_l.isAlive()) {
          TensorList inData = in_l.getData();
          @javax.annotation.Nonnull final TensorList tensorList = TensorArray.wrap(IntStream.range(0, inData.length()).parallel().mapToObj(dataIndex -> {
            Tensor deltaTensor = delta.get(dataIndex);
            final double deltaV = deltaTensor.get(0);
            deltaTensor.freeRef();
            @javax.annotation.Nonnull final Tensor passback = new Tensor(inData.getDimensions());
            final int dim = passback.dim();
            for (int i = 0; i < dim; i++) {
              passback.set(i, deltaV / dim);
            }
            return passback;
          }).toArray(i -> new Tensor[i]));
          in_l.accumulate(buffer, tensorList);
          tensorList.freeRef();
        }
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(ReferenceCounting::freeRef);
        Arrays.stream(inObj).map(NNResult::getData).forEach(ReferenceCounting::freeRef);
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
