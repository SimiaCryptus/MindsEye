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

package com.simiacryptus.mindseye.layers.loss;

import com.simiacryptus.mindseye.layers.DerivativeTester;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.util.ml.Tensor;

public class EntropyLossLayerTest extends LayerTestBase {
  
  @Override
  public NNLayer getLayer() {
    return new EntropyLossLayer();
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {4},{4}
    };
  }
  
  @Override
  public DerivativeTester getDerivativeTester() {
    return new DerivativeTester(1e-4, 1e-8) {
      @Override
      protected void testFeedback(NNLayer component, int i, Tensor outputPrototype, Tensor... inputPrototype) {
        if(i == 0) super.testFeedback(component, i, outputPrototype, inputPrototype);
      }
    };
  }
}
