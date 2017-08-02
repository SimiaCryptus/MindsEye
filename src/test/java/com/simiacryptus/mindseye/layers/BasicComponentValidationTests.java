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

import com.simiacryptus.mindseye.layers.activation.*;
import com.simiacryptus.mindseye.layers.cross.CrossDifferenceLayer;
import com.simiacryptus.mindseye.layers.cross.CrossProductLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.loss.StaticScalarLossLayer;
import com.simiacryptus.mindseye.layers.media.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.reducers.*;
import com.simiacryptus.mindseye.layers.synapse.*;
import com.simiacryptus.mindseye.layers.util.ConstNNLayer;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The type Basic component validation tests.
 */
public class BasicComponentValidationTests {
  
  private static final Logger log = LoggerFactory.getLogger(BasicComponentValidationTests.class);
  
  /**
   * Test const nn layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testConstNNLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final NNLayer component = new ConstNNLayer(new Tensor(outputPrototype.getDimensions()).map(i -> Util.R.get().nextGaussian()));
    ComponentTestUtil.test(component, outputPrototype);
  }
  
  /**
   * Test bias layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testBiasLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new BiasLayer(outputPrototype.getDimensions()).setWeights(i -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test img band bias layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testImgBandBiasLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2,2,3);
    final Tensor inputPrototype = new Tensor(2,2,3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgBandBiasLayer(outputPrototype.getDimensions()[2]).setWeights(i -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test dense synapse layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testDenseSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new JavaDenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDimensions()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test toeplitz synapse layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testToeplitzSynapseLayer1() throws Throwable {
    final Tensor inputPrototype = new Tensor(3, 3).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = new Tensor(3, 3);
    final NNLayer component = new ToeplitzSynapseLayer(inputPrototype.getDimensions(), outputPrototype.getDimensions()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test toeplitz synapse layer 2.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testToeplitzSynapseLayer2() throws Throwable {
    final Tensor inputPrototype = new Tensor(3, 3).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = new Tensor(3, 3, 2, 3);
    final NNLayer component = new ToeplitzSynapseLayer(inputPrototype.getDimensions(), outputPrototype.getDimensions()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test dense synapse layer jblas 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testDenseSynapseLayerJBLAS1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new DenseSynapseLayer(inputPrototype.getDimensions(), outputPrototype.getDimensions()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test transposed synapse layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testTransposedSynapseLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new TransposedSynapseLayer(new DenseSynapseLayer(inputPrototype.getDimensions(), outputPrototype.getDimensions()).setWeights(() -> Util.R.get().nextGaussian()));
    ComponentTestUtil.test(component, inputPrototype, outputPrototype);
  }
  
  /**
   * Test dense synapse layer 2.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testDenseSynapseLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new JavaDenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDimensions()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test entropy loss layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testEntropyLossLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextDouble());
    inputPrototype1 = inputPrototype1.scale(1. / inputPrototype1.l1());
    Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextDouble());
    inputPrototype2 = inputPrototype2.scale(1. / inputPrototype2.l1());
    final NNLayer component = new EntropyLossLayer();
    final Tensor[] inputPrototype = {inputPrototype1, inputPrototype2};
    ComponentTestUtil.testFeedback(component, 0, outputPrototype, inputPrototype);
    final int layers = component.state().size();
    for (int i = 0; i < layers; i++) {
      ComponentTestUtil.testLearning(component, i, outputPrototype, inputPrototype);
    }
  }
  
  /**
   * Test product layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testProductLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ProductLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  /**
   * Test re lu.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testReLu() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ReLuActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test linear.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testLinear() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new LinearActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test hyperbolic activation layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testHyperbolicActivationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(10);
    final Tensor inputPrototype = new Tensor(10).fill(() -> Util.R.get().nextGaussian());
    final HyperbolicActivationLayer component = new HyperbolicActivationLayer();
    component.setModeEven();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
    component.setModeOdd();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
    component.setModeAsymetric();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test sigmoid layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testSigmoidLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SigmoidActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test sigmoid layer 2.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testSigmoidLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SigmoidActivationLayer().setBalanced(false);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test softmax layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testSoftmaxLayer() throws Throwable {
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = inputPrototype.copy();
    final NNLayer component = new SoftmaxActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test sq activation layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testSqActivationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SqActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test log activation layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testLogActivationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new LogActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test sq loss layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testSqLossLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MeanSqLossLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  /**
   * Test static scalar loss layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testStaticScalarLossLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new StaticScalarLossLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1);
  }
  
  /**
   * Test cross product layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testCrossProductLayer() throws Throwable {
    Assert.assertEquals(0, CrossProductLayer.index(0,1, 4));
    Assert.assertEquals(3, CrossProductLayer.index(1,2, 4));
    Assert.assertEquals(5, CrossProductLayer.index(2,3, 4));
    final Tensor outputPrototype = new Tensor(6);
    final Tensor inputPrototype1 = new Tensor(4).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new CrossProductLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1);
  }
  
  /**
   * Test cross difference layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testCrossDifferenceLayer() throws Throwable {
    Assert.assertEquals(0, CrossDifferenceLayer.index(0,1, 4));
    Assert.assertEquals(3, CrossDifferenceLayer.index(1,2, 4));
    Assert.assertEquals(5, CrossDifferenceLayer.index(2,3, 4));
    final Tensor outputPrototype = new Tensor(6);
    final Tensor inputPrototype1 = new Tensor(4).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new CrossDifferenceLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1);
  }
  
  /**
   * Test sum layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testSumLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SumInputsLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  /**
   * Test product inputs layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testProductInputsLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ProductInputsLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  /**
   * Test sum reducer layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testSumReducerLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SumReducerLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  /**
   * Test avg reducer layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testAvgReducerLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new AvgReducerLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
}
