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
import com.simiacryptus.mindseye.layers.NNLayer;

import java.util.List;

public class VariableLayer extends NNLayerWrapper {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("inner", getInner().getJson());
    return json;
  }
  public static VariableLayer fromJson(JsonObject json) {
    return new VariableLayer(json);
  }
  protected VariableLayer(JsonObject json) {
    super(json);
  }
  
  public VariableLayer(final NNLayer inner) {
    super();
    setInner(inner);
  }

  @Override
  public List<NNLayer> getChildren() {
    return super.getChildren();
  }

  public final void setInner(final NNLayer inner) {
    this.inner = inner;
  }

}
