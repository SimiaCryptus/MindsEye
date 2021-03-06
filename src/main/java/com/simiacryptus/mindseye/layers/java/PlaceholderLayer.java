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

import com.google.common.hash.Hashing;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.Result;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * An ephemeral, non-serializable, non-evaluatable key. Used when a key is required as an identifier, e.g. DeltaSet
 *
 * @param <T> the type parameter
 */
@SuppressWarnings("serial")
public final class PlaceholderLayer<T> extends LayerBase {

  @Nullable
  private final T key;

  /**
   * Instantiates a new Placeholder key.
   *
   * @param key the key
   */
  public PlaceholderLayer(@Nullable final T key) {
    if (null == key) throw new UnsupportedOperationException();
    this.key = key;
    if (this.getKey() instanceof ReferenceCounting) {
      ((ReferenceCounting) this.getKey()).addRef();
    }
    setName(getClass().getSimpleName() + "/" + getId());
  }

  @Nonnull
  @Override
  public Result eval(final Result... array) {
    throw new UnsupportedOperationException();
  }

  @Nullable
  @Override
  public UUID getId() {
    T key = this.getKey();
    return key==null?UUID.randomUUID():UUID.fromString(key.toString());
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    throw new UnsupportedOperationException();
  }

  @Nonnull
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
  @Nullable
  public T getKey() {
    return key;
  }
}
