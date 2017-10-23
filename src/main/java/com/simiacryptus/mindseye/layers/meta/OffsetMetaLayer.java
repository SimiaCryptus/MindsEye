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
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Offset meta layer.
 */
@SuppressWarnings("serial")
public class OffsetMetaLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }

  /**
   * From json offset meta layer.
   *
   * @param json the json
   * @return the offset meta layer
   */
  public static OffsetMetaLayer fromJson(JsonObject json) {
    return new OffsetMetaLayer(json);
  }

  /**
   * Instantiates a new Offset meta layer.
   *
   * @param id the id
   */
  protected OffsetMetaLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(OffsetMetaLayer.class);
  
  /**
   * Instantiates a new Offset meta layer.
   */
  public OffsetMetaLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    int itemCnt = inObj[0].getData().length();
    double scale = inObj[1].getData().get(0).getData()[0];
    Tensor[] tensors = IntStream.range(0, itemCnt)
                         .parallel()
                         .mapToObj(dataIndex -> inObj[0].getData().get(dataIndex).map((v) -> v + scale))
                         .toArray(i -> new Tensor[i]);
    return new NNResult(tensors) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (inObj[0].isAlive()) {
          inObj[0].accumulate(buffer, new TensorArray(data.stream().map(t -> t.mapParallel((v) -> v)).toArray(i -> new Tensor[i])));
        }
        if (inObj[1].isAlive()) {
          double delta = tensors[0].mapCoordsParallel((v, c) -> {
            return IntStream.range(0, itemCnt).mapToDouble(i -> data.get(i).get(c)).sum();
          }).sum();
          Tensor passback = new Tensor(1);
          passback.set(0, delta);
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
