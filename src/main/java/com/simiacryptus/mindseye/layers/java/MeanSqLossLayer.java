/*
 * Copyright (c) 2017 by Andrew Charneski.
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
import java.util.stream.IntStream;

/**
 * The type Mean sq loss layer.
 */
public class MeanSqLossLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);
  
  /**
   * Instantiates a new Mean sq loss layer.
   *
   * @param id the id
   */
  protected MeanSqLossLayer(JsonObject id) {
    super(id);
  }
  
  /**
   * Instantiates a new Mean sq loss layer.
   */
  public MeanSqLossLayer() {
  }
  
  /**
   * From json mean sq loss layer.
   *
   * @param json the json
   * @return the mean sq loss layer
   */
  public static MeanSqLossLayer fromJson(JsonObject json) {
    return new MeanSqLossLayer(json);
  }
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    if (2 != inObj.length) throw new IllegalArgumentException();
    if (inObj[0].getData().length() != inObj[1].getData().length()) throw new IllegalArgumentException();
    Tensor diffs[] = new Tensor[inObj[0].getData().length()];
    Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
      final Tensor a = inObj[0].getData().get(dataIndex);
      final Tensor b = inObj[1].getData().get(dataIndex);
      if (a.dim() != b.dim()) {
        throw new IllegalArgumentException(String.format("%s != %s", Arrays.toString(a.getDimensions()), Arrays.toString(b.getDimensions())));
      }
      final Tensor r = a.minus(b);
      diffs[dataIndex] = r;
      return new Tensor(new double[]{r.sumSq() / r.dim()}, 1);
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (inObj[0].isAlive() || inObj[1].isAlive()) {
          Tensor[] passback = IntStream.range(0, data.length()).parallel().mapToObj(dataIndex -> {
            return diffs[dataIndex].scale(data.get(dataIndex).get(0) * 2.0 / diffs[dataIndex].dim());
          }).toArray(i -> new Tensor[i]);
          if (inObj[0].isAlive()) {
            inObj[0].accumulate(buffer, new TensorArray(passback));
          }
          if (inObj[1].isAlive()) {
            final Tensor[] negative = Arrays.stream(passback).map(x -> x.scale(-1)).toArray(i -> new Tensor[i]);
            inObj[1].accumulate(buffer, new TensorArray(negative));
          }
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
