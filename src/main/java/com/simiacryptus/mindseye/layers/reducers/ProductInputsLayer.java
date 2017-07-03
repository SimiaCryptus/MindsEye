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
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

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
      if(!Arrays.equals(inObj[0].data[0].getDims(), inObj[i].data[0].getDims()))
        throw new RuntimeException(Arrays.toString(inObj[0].data[0].getDims()) + " != " + Arrays.toString(inObj[i].data[0].getDims()));
    }
    Tensor[] result = Arrays.stream(inObj).map(x -> x.data).reduce((l, r) -> {
      return IntStream.range(0, Math.max(l.length,r.length))
                 .parallel()
                 .mapToObj(i->Tensor.product(l[Math.min(i,l.length-1)], r[Math.min(i,r.length-1)]))
                 .toArray(i->new Tensor[i]);
    }).get();
    return new NNResult(result) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
        assert Arrays.stream(delta).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        for (final NNResult input : inObj) {
          if (input.isAlive()) {
            input.accumulate(buffer, IntStream.range(0,input.data.length).mapToObj(i -> {
              return delta[Math.min(i,delta.length)].map((v, c)->{
                double v1 = input.data[i].get(c);
                double r = v * result[Math.min(i, result.length)].get(c) / v1;
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
