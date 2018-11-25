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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This key does not require any input, and produces a constant output. This constant can be tuned by optimization
 * processes.
 */
@SuppressWarnings("serial")
public class ValueLayer extends LayerBase {

  public static class RefWrapper<T> {
    public final T obj;

    public RefWrapper(T obj) {
      this.obj = obj;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      RefWrapper<?> that = (RefWrapper<?>) o;
      return obj == that.obj;
    }

    @Override
    public int hashCode() {
      return System.identityHashCode(obj);
    }
  }


  @Nullable
  private Tensor[] data;

  /**
   * Instantiates a new Const nn key.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ValueLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    JsonArray values = json.getAsJsonArray("values");
    data = IntStream.range(0,values.size()).mapToObj(i->Tensor.fromJson(values.get(i), resources)).toArray(i->new Tensor[i]);
  }

  /**
   * Instantiates a new Const nn key.
   *
   * @param data the data
   */
  public ValueLayer(final @Nonnull Tensor... data) {
    super();
    this.data = Arrays.copyOf(data,data.length);
    Arrays.stream(this.data)
        .map(x->new RefWrapper(x)).distinct().map(x->(Tensor)x.obj)
        .forEach(ReferenceCountingBase::addRef);
    this.frozen = true;
  }

  /**
   * From json const nn key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the const nn key
   */
  public static ValueLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ValueLayer(json, rs);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... array) {
    assert 0 == array.length;
    ValueLayer.this.addRef();
    Arrays.stream(data)
        .map(x->new RefWrapper(x)).distinct().map(x->(Tensor)x.obj)
        .forEach(ReferenceCountingBase::addRef);
    return new Result(TensorArray.create(ValueLayer.this.data), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      if (!isFrozen()) {
        assertAlive();
        assert (1 == ValueLayer.this.data.length || ValueLayer.this.data.length == data.length());
        for (int i = 0; i < data.length(); i++) {
          Tensor delta = data.get(i);
          Tensor value = ValueLayer.this.data[i % ValueLayer.this.data.length];
          buffer.get(value.getId(), value.getData()).addInPlace(delta.getData()).freeRef();
          delta.freeRef();
        }
      }
    }) {

      @Override
      protected void _free() {
        Arrays.stream(ValueLayer.this.data)
            .map(x->new RefWrapper(x)).distinct().map(x->(Tensor)x.obj)
            .forEach(ReferenceCountingBase::freeRef);
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

    Arrays.stream(ValueLayer.this.data)
        .map(x->new RefWrapper(x)).distinct().map(x->(Tensor)x.obj)
        .forEach(ReferenceCountingBase::freeRef);
  }

  /**
   * Gets data.
   *
   * @return the data
   */
  @Nullable
  public Tensor[] getData() {
    return data;
  }

  /**
   * Sets data.
   *
   * @param data the data
   */
  public void setData(final Tensor... data) {
    Arrays.stream(data)
        .map(x->new RefWrapper(x)).distinct().map(x->(Tensor)x.obj)
        .forEach(ReferenceCountingBase::addRef);
    if (null != this.data) Arrays.stream(this.data)
        .map(x->new RefWrapper(x)).distinct().map(x->(Tensor)x.obj)
        .forEach(ReferenceCountingBase::freeRef);
    this.data = data;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    JsonArray values = new JsonArray();
    Arrays.stream(data).map(datum -> datum.toJson(resources, dataSerializer)).forEach(values::add);
    json.add("values", values);
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.stream(data).map(x->x.getData()).collect(Collectors.toList());
  }
}
