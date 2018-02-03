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

import com.simiacryptus.mindseye.lang.NNLayer;

public class TestError extends RuntimeException {
  public final ComponentTest<?> test;
  public final NNLayer layer;
  
  public TestError(Throwable cause, ComponentTest<?> test, NNLayer layer) {
    super(String.format("Error in %s with %s", test, layer), cause);
    this.test = test;
    this.layer = layer;
    layer.setFloating(true);
  }
}
