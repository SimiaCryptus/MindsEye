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
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Map;

/**
 * An ephemeral, non-serializable, non-evaluatable layer. Used when a layer is required as an identifier, e.g. DeltaSet
 *
 * @param <T> the type parameter
 */
@SuppressWarnings("serial")
public final class PlaceholderLayer<T> extends NNLayer {
  
  private final @Nullable T key;
  
  /**
   * Instantiates a new Placeholder layer.
   *
   * @param key the key
   */
  public PlaceholderLayer(final @Nullable T key) {
    if (null == key) throw new UnsupportedOperationException();
    this.key = key;
    if (this.getKey() instanceof ReferenceCounting) {
      ((ReferenceCounting) this.getKey()).addRef();
    }
    setName(getClass().getSimpleName() + "/" + getId());
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(final NNResult... array) {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public @Nullable Object getId() {
    return this.getKey();
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    throw new UnsupportedOperationException();
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    throw new UnsupportedOperationException();
  }
  
  @Override
  protected void _free() {
    if (this.getKey() instanceof ReferenceCounting) {
      ((ReferenceCounting) this.getKey()).freeRef();
    }
    super._free();
  }
  
  /**
   * Gets key.
   *
   * @return the key
   */
  public @Nullable T getKey() {
    return key;
  }
}
