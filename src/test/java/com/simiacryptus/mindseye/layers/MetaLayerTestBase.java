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

package com.simiacryptus.mindseye.layers;

import com.simiacryptus.mindseye.test.BatchDerivativeTester;
import com.simiacryptus.mindseye.test.DerivativeTester;
import com.simiacryptus.mindseye.test.StandardLayerTests;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

/**
 * The type Meta layer test base.
 */
public abstract class MetaLayerTestBase extends StandardLayerTests {
  
  /**
   * Instantiates a new Meta layer test base.
   */
  public MetaLayerTestBase() {
    validateBatchExecution = false;
  }
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void test() throws Throwable {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      test(log);
    }
  }
  
  @Override
  public DerivativeTester getDerivativeTester() {
    return new BatchDerivativeTester(1e-3, 1e-4);
  }
}
