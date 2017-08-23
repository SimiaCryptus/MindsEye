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

package com.simiacryptus.mindseye.layers.reducers;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Avg reducer layer.
 */
public class AvgReducerLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }

  /**
   * From json avg reducer layer.
   *
   * @param json the json
   * @return the avg reducer layer
   */
  public static AvgReducerLayer fromJson(JsonObject json) {
    return new AvgReducerLayer(json);
  }

  /**
   * Instantiates a new Avg reducer layer.
   *
   * @param id the id
   */
  protected AvgReducerLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumReducerLayer.class);
  
  /**
   * Instantiates a new Avg reducer layer.
   */
  public AvgReducerLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    return new NNResult(IntStream.range(0, inObj[0].getData().length()).parallel().mapToDouble(dataIndex -> {
      double sum = 0;
      for (final NNResult element : inObj) {
        final double[] input = element.getData().get(dataIndex).getData();
        for (final double element2 : input) {
          sum += element2 / input.length;
        }
      }
      return sum;
    }).mapToObj(x->new Tensor(new int[]{1}, new double[]{x})).toArray(i->new Tensor[i])) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        for (final NNResult in_l : inObj) {
          if (in_l.isAlive()) {
            final Tensor[] data1 = IntStream.range(0, in_l.getData().length()).parallel().mapToObj(dataIndex -> {
              final double delta = data.get(dataIndex).get(0);
              final Tensor passback = new Tensor(in_l.getData().get(dataIndex).getDimensions());
              int dim = in_l.getData().get(dataIndex).dim();
              for (int i = 0; i < dim; i++) {
                passback.set(i, delta / dim);
              }
              return passback;
            }).toArray(i -> new Tensor[i]);
            in_l.accumulate(buffer, new TensorArray(data1));
          }
        }
      }
      
      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive())
            return true;
        return false;
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
