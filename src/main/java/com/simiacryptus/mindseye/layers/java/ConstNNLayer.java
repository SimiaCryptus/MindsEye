/*
 * Copyright (c) 2018 by Andrew Charneski.
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

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * This layer does not require any input, and produces a constant output. This constant can be tuned by optimization
 * processes.
 */
@SuppressWarnings("serial")
public class ConstNNLayer extends NNLayer {
  
  @Nullable
  private Tensor data;
  
  /**
   * Instantiates a new Const nn layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ConstNNLayer(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> resources) {
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
    this.frozen = true;
  }
  
  /**
   * From json const nn layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the const nn layer
   */
  public static ConstNNLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ConstNNLayer(json, rs);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... array) {
    Arrays.stream(array).forEach(nnResult -> nnResult.addRef());
    return new NNResult(TensorArray.create(data), (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      if (!isFrozen()) {
        data.stream().forEach(datum -> {
          buffer.get(ConstNNLayer.this, ConstNNLayer.this.data.getData()).addInPlace(datum.getData()).freeRef();
          datum.freeRef();
        });
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(array).forEach(nnResult -> nnResult.freeRef());
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
  @Nullable
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
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, @javax.annotation.Nonnull DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.add("value", data.toJson(resources, dataSerializer));
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(data.getData());
  }
}
