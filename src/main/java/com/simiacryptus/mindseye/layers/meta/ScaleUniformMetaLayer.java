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

package com.simiacryptus.mindseye.layers.meta;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.IntStream;

@SuppressWarnings("serial")
public class ScaleUniformMetaLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static ScaleUniformMetaLayer fromJson(JsonObject json) {
    return new ScaleUniformMetaLayer(UUID.fromString(json.get("id").getAsString()));
  }
  protected ScaleUniformMetaLayer(UUID id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ScaleUniformMetaLayer.class);
  
  public ScaleUniformMetaLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    double scale = inObj[1].data[0].getData()[0];
    Tensor[] tensors = IntStream.range(0, itemCnt).mapToObj(dataIndex -> inObj[0].data[dataIndex].map((v, c) -> v * scale)).toArray(i -> new Tensor[i]);
    return new NNResult(tensors) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (inObj[0].isAlive()) {
          inObj[0].accumulate(buffer, Arrays.stream(data).map(t -> t.map((v,c) -> v * scale)).toArray(i -> new Tensor[i]));
        }
        if (inObj[1].isAlive()) {
          double passback = tensors[0].map((v, c) -> {
            return IntStream.range(0, itemCnt).mapToDouble(i -> data[i].get(c) * inObj[0].data[i].get(c)).sum();
          }).sum();
          Tensor tensor = new Tensor(1);
          tensor.set(0,passback);
          inObj[1].accumulate(buffer, new Tensor[]{tensor});
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || inObj[1].isAlive();
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
