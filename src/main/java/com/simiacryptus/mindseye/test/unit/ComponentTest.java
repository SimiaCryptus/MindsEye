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
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nullable;
import java.util.Map;

/**
 * The interface Derivative tester.
 *
 * @param <T> the type parameter
 */
public interface ComponentTest<T> extends ReferenceCounting {
  /**
   * Test tolerance statistics.
   *
   * @param log            the log
   * @param component      the component
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Nullable
  T test(NotebookOutput log, Layer component, Tensor... inputPrototype);

}
