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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * This layer does not require any input, and produces a constant output. This constant can be tuned by optimization
 * processes.
 */
@SuppressWarnings("serial")
public class ValueLayer extends LayerBase {
  
  @Nullable
  private Tensor data;
  
  /**
   * Instantiates a new Const nn layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ValueLayer(@Nonnull final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    data = Tensor.fromJson(json.get("value"), resources);
  }
  
  /**
   * Instantiates a new Const nn layer.
   *
   * @param data the data
   */
  public ValueLayer(final Tensor data) {
    super();
    this.data = data;
    data.addRef();
    this.frozen = true;
  }
  
  /**
   * From json const nn layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the const nn layer
   */
  public static ValueLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ValueLayer(json, rs);
  }
  
  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... array) {
    assert 0 == array.length;
    ValueLayer.this.addRef();
    ValueLayer.this.data.addRef();
    return new Result(TensorArray.create(ValueLayer.this.data), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
      if (!isFrozen()) {
        data.stream().forEach(datum -> {
          buffer.get(ValueLayer.this, ValueLayer.this.data.getData()).addInPlace(datum.getData()).freeRef();
          datum.freeRef();
        });
      }
    }) {
      
      @Override
      protected void _free() {
        ValueLayer.this.data.freeRef();
        ValueLayer.this.freeRef();
      }
      
      @Override
      public boolean isAlive() {
        return !ValueLayer.this.isFrozen();
      }
    };
  }
  
  @Override
  protected void _free() {
    data.freeRef();
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
    data.addRef();
    if (null != this.data) this.data.freeRef();
    this.data = data;
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("value", data.toJson(resources, dataSerializer));
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(data.getData());
  }
}
