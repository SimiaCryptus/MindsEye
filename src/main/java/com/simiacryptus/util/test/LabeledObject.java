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

package com.simiacryptus.util.test;

import java.util.function.Function;

/**
 * The type Labeled object.
 *
 * @param <T> the type parameter
 */
public class LabeledObject<T> {
  /**
   * The Data.
   */
  public final T data;
  /**
   * The Label.
   */
  public final String label;
  
  /**
   * Instantiates a new Labeled object.
   *
   * @param img  the img
   * @param name the name
   */
  public LabeledObject(final T img, final String name) {
    super();
    this.data = img;
    this.label = name;
  }
  
  /**
   * Map labeled object.
   *
   * @param <U> the type parameter
   * @param f   the f
   * @return the labeled object
   */
  public <U> LabeledObject<U> map(final Function<T, U> f) {
    return new LabeledObject<>(f.apply(this.data), this.label);
  }
  
  @Override
  public String toString() {
    final StringBuffer sb = new StringBuffer("LabeledObject{");
    sb.append("data=").append(data);
    sb.append(", label='").append(label).append('\'');
    sb.append('}');
    return sb.toString();
  }
}
