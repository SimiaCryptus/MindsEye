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

import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * The type Binary entropy activation layer eval.
 */
public abstract class BinaryEntropyActivationLayerTest extends ActivationLayerTestBase {
  /**
   * Instantiates a new Binary entropy activation layer eval.
   */
  public BinaryEntropyActivationLayerTest() {
    super(new BinaryEntropyActivationLayer());
  }
  
  @Override
  public double random() {
    return 0.1 * Math.random() + 1.0;
  }
  
  @Override
  public DoubleStream scan() {
    return IntStream.range(50, 450).mapToDouble(x -> x * 1.0 / 500.0);
  }

//  /**
//   * Basic Test
//   */
//  public static class Basic extends BinaryEntropyActivationLayerTest {
//  }
  
}
