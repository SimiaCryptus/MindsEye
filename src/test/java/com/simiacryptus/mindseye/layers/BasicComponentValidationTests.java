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
import com.simiacryptus.mindseye.layers.media.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.reducers.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.reducers.ProductLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.layers.reducers.SumReducerLayer;
import com.simiacryptus.mindseye.layers.synapse.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BasicComponentValidationTests {
  
  private static final Logger log = LoggerFactory.getLogger(BasicComponentValidationTests.class);
  
  @Test
  public void testBiasLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new BiasLayer(outputPrototype.getDims()).setWeights(i -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testImgBandBiasLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2,2,3);
    final Tensor inputPrototype = new Tensor(2,2,3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgBandBiasLayer(outputPrototype.getDims()[2]).setWeights(i -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testDenseSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new JavaDenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testToeplitzSynapseLayer1() throws Throwable {
    final Tensor inputPrototype = new Tensor(3, 3).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = new Tensor(3, 3);
    final NNLayer component = new ToeplitzSynapseLayer(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testToeplitzSynapseLayer2() throws Throwable {
    final Tensor inputPrototype = new Tensor(3, 3).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = new Tensor(3, 3, 2, 3);
    final NNLayer component = new ToeplitzSynapseLayer(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testDenseSynapseLayerJBLAS1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new DenseSynapseLayer(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testTransposedSynapseLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new TransposedSynapseLayer(new DenseSynapseLayer(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian()));
    ComponentTestUtil.test(component, inputPrototype, outputPrototype);
  }
  
  @Test
  public void testDenseSynapseLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new JavaDenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
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
  
  @Test
  public void testProductLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ProductLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  @Test
  public void testReLu() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ReLuActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testLinear() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new LinearActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testHyperbolicActivationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new HyperbolicActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testSigmoidLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SigmoidActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testSigmoidLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SigmoidActivationLayer().setBalanced(false);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testSoftmaxLayer() throws Throwable {
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = inputPrototype.copy();
    final NNLayer component = new SoftmaxActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testSqActivationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SqActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testLogActivationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new LogActivationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testSqLossLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MeanSqLossLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
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
  
  @Test
  public void testSumLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SumInputsLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  @Test
  public void testSumReducerLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SumReducerLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  @Test
  public void testAvgReducerLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new AvgReducerLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
}
