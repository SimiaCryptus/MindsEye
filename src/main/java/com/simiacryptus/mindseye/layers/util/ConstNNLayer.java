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

package com.simiacryptus.mindseye.layers.util;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.TensorList;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * The type Const nn layer.
 */
public class ConstNNLayer extends NNLayer {
  
  @Override
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("value", tensor.getJson());
    return json;
  }
  
  /**
   * From json const nn layer.
   *
   * @param json the json
   * @return the const nn layer
   */
  public static ConstNNLayer fromJson(JsonObject json) {
    return new ConstNNLayer(json);
  }

  /**
   * Instantiates a new Const nn layer.
   *
   * @param json the json
   */
  protected ConstNNLayer(JsonObject json) {
    super(json);
    this.tensor = Tensor.fromJson(json.getAsJsonObject("value"));
  }
  
  /**
   * The Tensor.
   */
  Tensor tensor;
  
  /**
   * Instantiates a new Const nn layer.
   *
   * @param tensor the tensor
   */
  public ConstNNLayer(Tensor tensor) {
    super();
    this.tensor = tensor;
    setFrozen(true);
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, NNResult... array) {
    return new NNResult(tensor) {
      @Override
      public void accumulate(DeltaSet buffer, TensorList data) {
        if(!isFrozen()) {
          data.stream().forEach(datum->{
            buffer.get(ConstNNLayer.this, tensor).accumulate(datum.getData());
          });
        }
      }
      
      @Override
      public boolean isAlive() {
        return !ConstNNLayer.this.isFrozen();
      }
    };
  }
  
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(tensor.getData());
  }
}
