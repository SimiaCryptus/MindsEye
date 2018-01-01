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

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * This layer does not require any input, and produces a constant output. This constant can be tuned by optimization
 * processes.
 */
@SuppressWarnings("serial")
public class ConstNNLayer extends NNLayer {
  
  private Tensor data;
  
  /**
   * Instantiates a new Const nn layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ConstNNLayer(final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    data = Tensor.fromJson(json.get("value"), resources);
  }
  
  /**
   * Instantiates a new Const nn layer.
   *
   * @param data the data
   */
  public ConstNNLayer(final Tensor data) {
    super();
    this.data = data;
    setFrozen(true);
  }
  
  /**
   * From json const nn layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the const nn layer
   */
  public static ConstNNLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ConstNNLayer(json, rs);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... array) {
    return new NNResult(data) {
  
      @Override
      public void finalize() {
        Arrays.stream(array).forEach(NNResult::finalize);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        if (!isFrozen()) {
          data.stream().forEach(datum -> {
            buffer.get(ConstNNLayer.this, ConstNNLayer.this.data.getData()).addInPlace(datum.getData());
          });
        }
      }
      
      @Override
      public boolean isAlive() {
        return !ConstNNLayer.this.isFrozen();
      }
    };
  }
  
  /**
   * Gets data.
   *
   * @return the data
   */
  public Tensor getData() {
    return data;
  }
  
  /**
   * Sets data.
   *
   * @param data the data
   */
  public void setData(final Tensor data) {
    this.data = data;
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("value", data.toJson(resources, dataSerializer));
    return json;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(data.getData());
  }
}
