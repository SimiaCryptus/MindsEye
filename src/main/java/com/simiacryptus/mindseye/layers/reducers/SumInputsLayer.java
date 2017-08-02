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

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Sum inputs layer.
 */
public class SumInputsLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }

  /**
   * From json sum inputs layer.
   *
   * @param json the json
   * @return the sum inputs layer
   */
  public static SumInputsLayer fromJson(JsonObject json) {
    return new SumInputsLayer(json);
  }

  /**
   * Instantiates a new Sum inputs layer.
   *
   * @param id the id
   */
  protected SumInputsLayer(JsonObject id) {
    super(id);
  }

  /**
   * Instantiates a new Sum inputs layer.
   */
  public SumInputsLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    TensorList data = Arrays.stream(inObj).map(x -> x.data).reduce((l, r) -> {
      return new TensorArray(IntStream.range(0, l.length())
              .parallel()
              .mapToObj(i->Tensor.add(l.get(i), r.get(i)))
              .toArray(i->new Tensor[i]));
    }).get();
    return new NNResult(data) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        assert data.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        for (final NNResult input : inObj) {
          if (input.isAlive()) {
            input.accumulate(buffer, data);
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
