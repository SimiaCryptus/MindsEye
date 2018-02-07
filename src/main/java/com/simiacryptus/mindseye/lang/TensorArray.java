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

package com.simiacryptus.mindseye.lang;

import java.util.Arrays;
import java.util.stream.Stream;

/**
 * An on-heap implementation of the TensorList data container.
 */
public class TensorArray extends ReferenceCountingBase implements TensorList {
  @javax.annotation.Nonnull
  private final Tensor[] data;
  
  /**
   * Instantiates a new Tensor array.
   *
   * @param data the data
   */
  private TensorArray(@javax.annotation.Nonnull final Tensor... data) {
    this.data = data;
    for (@javax.annotation.Nonnull Tensor tensor : data) {
      tensor.addRef();
    }
  }
  
  /**
   * Create tensor array.
   *
   * @param data the data
   * @return the tensor array
   */
  public static TensorArray create(final Tensor... data) {
    return new TensorArray(data);
  }
  
  /**
   * Wrap tensor array.
   *
   * @param data the data
   * @return the tensor array
   */
  @javax.annotation.Nonnull
  public static TensorArray wrap(@javax.annotation.Nonnull final Tensor... data) {
    @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.create(data);
    for (@javax.annotation.Nonnull Tensor tensor : data) {
      tensor.freeRef();
    }
    return tensorArray;
  }
  
  /**
   * To string string.
   *
   * @param <T>   the type parameter
   * @param limit the limit
   * @param data  the data
   * @return the string
   */
  public static <T> String toString(int limit, @javax.annotation.Nonnull T... data) {
    return (data.length < limit) ? Arrays.toString(data) : "[" + Arrays.stream(data).limit(limit).map(x -> x.toString()).reduce((a, b) -> a + ", " + b).get() + ", ...]";
  }
  
  @Override
  public Tensor get(final int i) {
    Tensor datum = data[i];
    datum.addRef();
    return datum;
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[] getDimensions() {
    return data[0].getDimensions();
  }
  
  @Override
  public int length() {
    return data.length;
  }
  
  @javax.annotation.Nonnull
  @Override
  public Stream<Tensor> stream() {
    return Arrays.stream(data);
  }
  
  @Override
  public String toString() {
    return String.format("TensorArray{data=%s}", toString(9, data));
  }
  
  @Override
  protected void _free() {
    try {
      for (@javax.annotation.Nonnull final Tensor d : data) {
        d.freeRef();
      }
    } catch (@javax.annotation.Nonnull final RuntimeException e) {
      throw e;
    } catch (@javax.annotation.Nonnull final Throwable e) {
      throw new RuntimeException(e);
    }
  }
}
