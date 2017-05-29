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
import com.simiacryptus.mindseye.layers.activation.L1NormalizationLayer;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.IntStream;

public class SumInputsLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static SumInputsLayer fromJson(JsonObject json) {
    return new SumInputsLayer(UUID.fromString(json.get("id").getAsString()));
  }
  protected SumInputsLayer(UUID id) {
    super(id);
  }

  public SumInputsLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    Tensor[] data = Arrays.stream(inObj).map(x -> x.data).reduce((l, r) -> {
      return IntStream.range(0, l.length)
                 .parallel()
                 .mapToObj(i->Tensor.add(l[i], r[i]))
                 .toArray(i->new Tensor[i]);
    }).get();
    return new NNResult(data) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
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
