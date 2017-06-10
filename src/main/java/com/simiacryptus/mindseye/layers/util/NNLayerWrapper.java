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
import com.simiacryptus.util.ScalarStatistics;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class NNLayerWrapper extends NNLayer {
  public final NNLayer inner;
  
  public NNLayerWrapper(JsonObject json) {
    super(json);
    this.inner = fromJson(json.getAsJsonObject("inner"));
  }
  
  public NNLayerWrapper(NNLayer inner) {
    this.inner = inner;
  }
  
  protected NNLayerWrapper() {
    this.inner = null;
  }
  
  @Override
  public List<double[]> state() {
    return this.inner.state();
  }
  
  @Override
  public String getName() {
    return inner.getName();
  }
  
  @Override
  public NNLayer setName(String name) {
    if(null != inner) inner.setName(name);
    return this;
  }
  
  @Override
  public NNLayer freeze() {
    inner.freeze();
    return this;
  }
  
  @Override
  public boolean isFrozen() {
    if(null == inner) return true;
    return inner.isFrozen();
  }
  
  @Override
  public NNLayer setFrozen(boolean frozen) {
    if(inner!=null) inner.setFrozen(frozen);
    return this;
  }
}
