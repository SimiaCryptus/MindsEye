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
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;

import java.util.List;
import java.util.Map;

/**
 * A base class for layers whose actual behavior is delegated.
 */
@SuppressWarnings("serial")
public abstract class WrapperLayer extends NNLayer {
  /**
   * The Inner.
   */
  protected NNLayer inner;
  
  /**
   * Instantiates a new Wrapper layer.
   */
  protected WrapperLayer() {
    inner = null;
  }
  
  /**
   * Instantiates a new Wrapper layer.
   *
   * @param json the json
   */
  public WrapperLayer(final JsonObject json) {
    super(json);
    inner = NNLayer.fromJson(json.getAsJsonObject("inner"));
  }
  
  /**
   * Instantiates a new Wrapper layer.
   *
   * @param inner the inner
   */
  public WrapperLayer(final NNLayer inner) {
    this.inner = inner;
  }
  
  @Override
  public NNResult eval(final NNResult... array) {
    return inner.eval(array);
  }
  
  /**
   * Gets inner.
   *
   * @return the inner
   */
  public final NNLayer getInner() {
    return inner;
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("inner", getInner().getJson(resources, dataSerializer));
    return json;
  }
  
  @Override
  public boolean isFrozen() {
    if (null == inner) return true;
    return inner.isFrozen();
  }
  
  @Override
  public NNLayer setFrozen(final boolean frozen) {
    if (inner != null) {
      inner.setFrozen(frozen);
    }
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return inner.state();
  }
}
