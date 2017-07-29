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
import java.util.stream.Stream;

public class ProductInputsLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static ProductInputsLayer fromJson(JsonObject json) {
    return new ProductInputsLayer(json);
  }
  protected ProductInputsLayer(JsonObject id) {
    super(id);
  }

  public ProductInputsLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    assert inObj.length > 1;
    for(int i=1;i<inObj.length;i++) {
      if(!Arrays.equals(inObj[0].data.get(0).getDimensions(), inObj[i].data.get(0).getDimensions()))
        throw new RuntimeException(Arrays.toString(inObj[0].data.get(0).getDimensions()) + " != " + Arrays.toString(inObj[i].data.get(0).getDimensions()));
    }
    TensorList result = Arrays.stream(inObj).parallel().map(x -> x.data).reduce((l, r) -> {
      Stream<Tensor> tensorStream = IntStream.range(0, Math.max(l.length(), r.length())).parallel()
              .mapToObj(i -> Tensor.product(l.get(Math.min(i, l.length() - 1)), r.get(Math.min(i, r.length() - 1))));
      return new TensorArray(tensorStream.toArray(i->new Tensor[i]));
    }).get();
    return new NNResult(result) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
        assert Arrays.stream(delta).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        for (final NNResult input : inObj) {
          if (input.isAlive()) {
            input.accumulate(buffer, IntStream.range(0, input.data.length()).parallel().mapToObj(i -> {
              return delta[Math.min(i,delta.length)].map((v, c)->{
                double v1 = input.data.get(i).get(c);
                double r = v * result.get(Math.min(i, result.length())).get(c) / v1;
                return Double.isFinite(r)?r:0.0;
              });
            }).toArray(i->new Tensor[i]));
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
