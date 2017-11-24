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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.ActivationLayerTestBase;
import com.simiacryptus.util.Util;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * The type Layer run base.
 */
public abstract class LayerTestBase {
  private static final Logger log = LoggerFactory.getLogger(ActivationLayerTestBase.class);
  
  /**
   * Test derivatives.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testDerivatives() throws Throwable {
    Tensor[] inputPrototype = Arrays.stream(getInputDims()).map(dim -> new Tensor(dim).fill(() -> Util.R.get().nextDouble()))
      .toArray(i -> new Tensor[i]);
    Tensor outputPrototype = GpuController.INSTANCE.distribute(Arrays.<Tensor[]>asList(inputPrototype),
      (data, exe) -> getLayer().eval(exe, NNResult.batchResultArray(data.toArray(new Tensor[][]{}))).getData().get(0),
      (a, b) -> a.add(b));
    getDerivativeTester().test(getLayer(), outputPrototype, inputPrototype);
    getPerformanceTester().test(getLayer(), outputPrototype, inputPrototype);
    getEquivalencyTester().test(getReferenceLayer(), getLayer(), outputPrototype, inputPrototype);
  }
  
  public EquivalencyTester getEquivalencyTester() {
    return new EquivalencyTester(1e-5);
  }
  
  public PerformanceTester getPerformanceTester() {
    return new PerformanceTester();
  }
  
  /**
   * Test json.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testJson() throws Throwable {
    NNLayer layer = getLayer();
    NNLayer echo = NNLayer.fromJson(layer.getJson());
    assert (echo != null && layer != echo);
  }
  
  /**
   * Gets derivative tester.
   *
   * @return the derivative tester
   */
  public DerivativeTester getDerivativeTester() {
    return new DerivativeTester(1e-10, 1e-8);
  }
  
  /**
   * Gets layer.
   *
   * @return the layer
   */
  public abstract NNLayer getLayer();
  
  public NNLayer getReferenceLayer() {
    return null;
  }
  
  /**
   * Get input dims int [ ] [ ].
   *
   * @return the int [ ] [ ]
   */
  public abstract int[][] getInputDims();
}
