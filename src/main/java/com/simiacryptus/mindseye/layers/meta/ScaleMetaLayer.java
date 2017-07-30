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
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

@SuppressWarnings("serial")
public class ScaleMetaLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static ScaleMetaLayer fromJson(JsonObject json) {
    return new ScaleMetaLayer(json);
  }
  protected ScaleMetaLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ScaleMetaLayer.class);
  
  public ScaleMetaLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length();
    Tensor[] tensors = IntStream.range(0, itemCnt).mapToObj(dataIndex -> inObj[0].data.get(dataIndex).map((v,c)->v * inObj[1].data.get(0).get(c))).toArray(i -> new Tensor[i]);
    return new NNResult(tensors) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (inObj[0].isAlive()) {
          inObj[0].accumulate(buffer, new TensorArray(data.stream().map(t -> t.map((v,c) -> v * inObj[1].data.get(0).get(c))).toArray(i -> new Tensor[i])));
        }
        if (inObj[1].isAlive()) {
          Tensor passback = tensors[0].map((v, c) -> {
            return IntStream.range(0, itemCnt).mapToDouble(i -> data.get(i).get(c) * inObj[0].data.get(i).get(c)).sum();
          });
          inObj[1].accumulate(buffer, new TensorArray(passback));
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
