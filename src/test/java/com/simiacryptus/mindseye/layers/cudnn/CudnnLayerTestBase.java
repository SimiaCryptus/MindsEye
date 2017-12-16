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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.PrintStream;

/**
 * The type Cudnn layer test base.
 */
public abstract class CudnnLayerTestBase extends LayerTestBase {
  
  /**
   * Instantiates a new Cudnn layer test base.
   */
  public CudnnLayerTestBase() {
  }
  
  @Override
  protected ComponentTest getReferenceIOTester() {
    ComponentTest inner = super.getReferenceIOTester();
    return new ComponentTest() {
      @Override
      public ToleranceStatistics test(NotebookOutput log, NNLayer component, Tensor... inputPrototype) {
        try {
          CuDNN.apiLog = new PrintStream(log.file("cuda.log"));
          return inner.test(log, component, inputPrototype);
        } finally {
          log.p(log.file(null, "cuda.log", "GPU Log"));
          CuDNN.apiLog.close();
          CuDNN.apiLog = null;
        }
      }
    };
  }
  
}
