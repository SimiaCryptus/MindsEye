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

package com.simiacryptus.mindseye.test.unit;

import com.simiacryptus.mindseye.lang.Layer;

import javax.annotation.Nonnull;

/**
 * The type Test error.
 */
public class TestError extends RuntimeException {
  /**
   * The Test.
   */
  public final ComponentTest<?> test;
  /**
   * The LayerBase.
   */
  @Nonnull
  public final Layer layer;

  /**
   * Instantiates a new Test error.
   *
   * @param cause the cause
   * @param test  the test
   * @param layer the key
   */
  public TestError(Throwable cause, ComponentTest<?> test, @Nonnull Layer layer) {
    super(String.format("Error in %s apply %s", test, layer), cause);
    this.test = test;
    this.test.addRef();
    this.layer = layer;
    this.layer.addRef();
    layer.detach();
  }
}
