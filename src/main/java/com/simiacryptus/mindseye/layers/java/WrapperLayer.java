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
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.List;
import java.util.Map;

/**
 * A base class for layers whose actual behavior is delegated.
 */
@SuppressWarnings("serial")
public abstract class WrapperLayer extends LayerBase {
  @Nullable
  private Layer inner;

  /**
   * Instantiates a new Wrapper key.
   */
  protected WrapperLayer() {
    inner = null;
  }

  /**
   * Instantiates a new Wrapper key.
   *
   * @param json the json
   * @param rs   the rs
   */
  public WrapperLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    this.inner = Layer.fromJson(json.getAsJsonObject("inner"), rs);
  }

  /**
   * Instantiates a new Wrapper key.
   *
   * @param inner the heapCopy
   */
  public WrapperLayer(final Layer inner) {
    this.inner = inner;
    this.inner.addRef();
  }

  @Override
  protected void _free() {
    if (null != this.inner) this.inner.freeRef();
    super._free();
  }

  @Nullable
  @Override
  public Result eval(final Result... array) {
    return inner.eval(array);
  }

  /**
   * The Inner.
   */
  /**
   * Gets heapCopy.
   *
   * @return the heapCopy
   */
  @Nullable
  public final Layer getInner() {
    return inner;
  }

  /**
   * Sets inner.
   *
   * @param inner the inner
   * @return the inner
   */
  public WrapperLayer setInner(@Nullable Layer inner) {
    this.inner = inner;
    return this;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("inner", getInner().getJson(resources, dataSerializer));
    return json;
  }

  @Override
  public boolean isFrozen() {
    if (null == inner) return true;
    return inner.isFrozen();
  }

  @Nonnull
  @Override
  public Layer setFrozen(final boolean frozen) {
    if (null == inner) return this;
    inner.setFrozen(frozen);
    return this;
  }

  @Nullable
  @Override
  public List<double[]> state() {
    return inner.state();
  }
}
