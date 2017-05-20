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

package com.simiacryptus.mindseye.net;

import com.simiacryptus.mindseye.graph.PipelineNetwork;
import com.simiacryptus.mindseye.net.activation.L1NormalizationLayer;
import com.simiacryptus.mindseye.net.activation.MaxConstLayer;
import com.simiacryptus.mindseye.net.activation.MaxDropoutNoiseLayer;
import com.simiacryptus.mindseye.net.media.ImgConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.activation.EntropyLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.net.media.SumSubsampleLayer;
import com.simiacryptus.mindseye.net.reducers.ImgConcatLayer;
import com.simiacryptus.mindseye.net.reducers.SumInputsLayer;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MediaComponentValidationTests {
  public static final double deltaFactor = 1e-5;
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MediaComponentValidationTests.class);
  
  @Test
  public void testImgConcatLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2,3,5);
    final Tensor inputPrototype1 = new Tensor(2,3,1).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2,3,4).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgConcatLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  @Test
  public void testConvolutionSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(3, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 3, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgConvolutionSynapseLayer(3, 3, 4).addWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testConvolutionSynapseLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 2, 1);
    final Tensor inputPrototype = new Tensor(2, 3, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgConvolutionSynapseLayer(2, 2, 1).addWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testConvolutionSynapseLayer3() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 2);
    final Tensor inputPrototype = new Tensor(1, 1, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgConvolutionSynapseLayer(1, 1, 4).addWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testConvolutionSynapseLayer4() throws Throwable {
    final Tensor outputPrototype = new Tensor(2, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 5, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgConvolutionSynapseLayer(2, 3, 4).addWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testL1NormalizationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new L1NormalizationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testMaxConstLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MaxConstLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testMaxEntLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextDouble());
    final PipelineNetwork component = new PipelineNetwork();
    component.add(new L1NormalizationLayer());
    component.add(new EntropyLayer());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1);
  }
  
  @Test
  public void testMaxSubsampleLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MaxSubsampleLayer(2, 2, 1);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testMaxDropoutNoiseLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2, 2, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MaxDropoutNoiseLayer(2, 2, 1);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testSumSubsampleLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SumSubsampleLayer(2, 2, 1);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  @Test
  public void testSumSubsampleLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 2);
    final Tensor inputPrototype = new Tensor(3, 5, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new SumSubsampleLayer(3, 5, 1);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
}
