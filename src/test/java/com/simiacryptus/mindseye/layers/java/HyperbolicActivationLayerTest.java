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

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;

import java.util.HashMap;

/**
 * The type Hyperbolic activation layer test.
 */
public abstract class HyperbolicActivationLayerTest extends ActivationLayerTestBase {
  /**
   * Instantiates a new Hyperbolic activation layer test.
   */
  public HyperbolicActivationLayerTest() {
    super(new HyperbolicActivationLayer());
  }
  
  @Override
  protected HashMap<Tensor[], Tensor> getReferenceIO() {
    final HashMap<Tensor[], Tensor> map = super.getReferenceIO();
    map.put(new Tensor[]{new Tensor(0.0)}, new Tensor(0.0));
    return map;
  }
  
  @Override
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return new TrainingTester().setRandomizationMode(TrainingTester.RandomizationMode.Random);
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends HyperbolicActivationLayerTest {
  }
  
}
