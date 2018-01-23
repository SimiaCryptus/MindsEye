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

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Computes the average value across all elements of each input tensor. The output dimensions are always 1x1x1.
 */
@SuppressWarnings("serial")
public class AvgReducerLayer extends NNLayer {
  
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
  protected AvgReducerLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json avg reducer layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the avg reducer layer
   */
  public static AvgReducerLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new AvgReducerLayer(json);
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    return new NNResult(IntStream.range(0, inObj[0].getData().length()).parallel().mapToDouble(dataIndex -> {
      double sum = 0;
      for (final NNResult element : inObj) {
        final double[] input = element.getData().get(dataIndex).getData();
        for (final double element2 : input) {
          sum += element2 / input.length;
        }
      }
      return sum;
    }).mapToObj(x -> new Tensor(new double[]{x}, new int[]{1})).toArray(i -> new Tensor[i])) {
  
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      protected void _accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        for (final NNResult in_l : inObj) {
          if (in_l.isAlive()) {
            final TensorList tensorList = new TensorArray(IntStream.range(0, in_l.getData().length()).parallel().mapToObj(dataIndex -> {
              final double delta = data.get(dataIndex).get(0);
              final Tensor passback = new Tensor(in_l.getData().get(dataIndex).getDimensions());
              final int dim = in_l.getData().get(dataIndex).dim();
              for (int i = 0; i < dim; i++) {
                passback.set(i, delta / dim);
              }
              return passback;
            }).toArray(i -> new Tensor[i]));
            in_l.accumulate(buffer, tensorList);
            tensorList.freeRef();
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
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
