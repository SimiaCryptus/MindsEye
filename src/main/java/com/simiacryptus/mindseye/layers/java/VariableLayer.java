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
import com.simiacryptus.mindseye.lang.Layer;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Map;

/**
 * Acts as a mutable placeholder layer, whose heapCopy implementation can be setByCoord and changed.
 */
@SuppressWarnings("serial")
public class VariableLayer extends WrapperLayer {
  
  /**
   * Instantiates a new Variable layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected VariableLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }
  
  /**
   * Instantiates a new Variable layer.
   *
   * @param inner the heapCopy
   */
  public VariableLayer(final Layer inner) {
    super(inner);
  }
  
  /**
   * From json variable layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the variable layer
   */
  public static VariableLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new VariableLayer(json, rs);
  }
  
  @Override
  public List<Layer> getChildren() {
    return super.getChildren();
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJson(resources, dataSerializer);
    json.add("heapCopy", getInner().getJson(resources, dataSerializer));
    return json;
  }
  
  /**
   * Sets heapCopy.
   *
   * @param inner the heapCopy
   */
  public final WrapperLayer setInner(final Layer inner) {
    if (this.getInner() != null) this.getInner().freeRef();
    super.setInner(inner);
    this.getInner().addRef();
    return null;
  }
  
}
