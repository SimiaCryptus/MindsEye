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

import com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.PoolingLayer;
import com.simiacryptus.mindseye.layers.media.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.layers.activation.L1NormalizationLayer;
import com.simiacryptus.mindseye.layers.activation.MaxConstLayer;
import com.simiacryptus.mindseye.layers.activation.MaxDropoutNoiseLayer;
import com.simiacryptus.mindseye.layers.activation.EntropyLayer;
import com.simiacryptus.mindseye.layers.reducers.ImgConcatLayer;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.IntStream;

/**
 * The type Media component validation tests.
 */
public class MediaComponentValidationTests {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MediaComponentValidationTests.class);
  
  /**
   * Test img concat layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testImgConcatLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2,3,5);
    final Tensor inputPrototype1 = new Tensor(2,3,1).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2,3,4).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgConcatLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }
  
  /**
   * Test convolution synapse layer stress.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testConvolutionSynapseLayerStress() throws Throwable {
    final NNLayer component = new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(3, 3, 4).addWeights(() -> Util.R.get().nextGaussian());
    component.eval(new NNLayer.NNExecutionContext() {}, NNResult.batchResultArray(IntStream.range(0,1000).mapToObj(i->{
      return new Tensor(3, 3, 2).fill(() -> Util.R.get().nextGaussian());
    }).map(i->new Tensor[]{i}).toArray(i->new Tensor[i][])));
  }
  
  /**
   * Test img reshape layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testImgCropLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(4, 4, 2);
    final Tensor inputPrototype = new Tensor(6, 6, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgCropLayer(2, 2);
    ComponentTestUtil.tolerance = 1e-4;
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test img reshape layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testImgReshapeLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(6, 6, 1);
    final Tensor inputPrototype = new Tensor(3, 3, 4).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgReshapeLayer(2, 2, true);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test img reshape layer 2.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testImgReshapeLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(3, 3, 4);
    final Tensor inputPrototype = new Tensor(6, 6, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgReshapeLayer(2, 2, false);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  /**
   * Test cu dnn convolution synapse layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testCuDNNConvolutionSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(3, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 3, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer(3, 3, 4)
            .addWeights(() -> Util.R.get().nextGaussian());
    double prev = ComponentTestUtil.tolerance;
    ComponentTestUtil.tolerance = 1e-4;
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
    ComponentTestUtil.tolerance = prev;
  }

  /**
   * Test cu dnn direct convolution synapse layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testCuDNNDirectConvolutionSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(3, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 3, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer(3, 3, 4)
            .addWeights(() -> Util.R.get().nextGaussian());
    double prev = ComponentTestUtil.tolerance;
    ComponentTestUtil.tolerance = 1e-4;
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
    ComponentTestUtil.tolerance = prev;
  }

  /**
   * Test cu dnn direct float convolution synapse layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testCuDNNDirectFloatConvolutionSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(3, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 3, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ConvolutionLayer(3, 3, 4)
            .addWeights(() -> Util.R.get().nextGaussian());
    double prev = ComponentTestUtil.tolerance;
    ComponentTestUtil.tolerance = 1e-4;
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
    ComponentTestUtil.tolerance = prev;
  }

  /**
   * Test cu dnn direct img bias layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testCuDNNDirectImgBiasLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(3, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 3, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ImgBandBiasLayer(2)
            .addWeights(() -> Util.R.get().nextGaussian());
    double prev = ComponentTestUtil.tolerance;
    ComponentTestUtil.tolerance = 1e-4;
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
    ComponentTestUtil.tolerance = prev;
  }

  /**
   * Test cu dnn direct activation layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testCuDNNDirectActivationLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(3, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 3, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ActivationLayer(ActivationLayer.Mode.RELU);
    double prev = ComponentTestUtil.tolerance;
    ComponentTestUtil.tolerance = 1e-4;
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
    ComponentTestUtil.tolerance = prev;
  }

  /**
   * Test cu dnn direct pooling layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testCuDNNDirectPoolingLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2, 2, 2);
    final Tensor inputPrototype = new Tensor(4, 4, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new PoolingLayer();
    double prev = ComponentTestUtil.tolerance;
    ComponentTestUtil.tolerance = 1e-4;
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
    ComponentTestUtil.tolerance = prev;
  }

  /**
   * Test cu dnn direct activation layer 2.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testCuDNNDirectActivationLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(3, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 3, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new ActivationLayer(ActivationLayer.Mode.SIGMOID);
    double prev = ComponentTestUtil.tolerance;
    ComponentTestUtil.tolerance = 1e-4;
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
    ComponentTestUtil.tolerance = prev;
  }

  /**
   * Test convolution synapse layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testConvolutionSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(3, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 3, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(3, 3, 4).addWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }

  /**
   * Test convolution synapse layer 2.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testConvolutionSynapseLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 2, 1);
    final Tensor inputPrototype = new Tensor(2, 3, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(2, 2, 1, false).addWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test convolution synapse layer 3.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testConvolutionSynapseLayer3() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 2);
    final Tensor inputPrototype = new Tensor(1, 1, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(1, 1, 4).addWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test convolution synapse layer 4.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testConvolutionSynapseLayer4() throws Throwable {
    final Tensor outputPrototype = new Tensor(2, 3, 2);
    final Tensor inputPrototype = new Tensor(3, 5, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer(2, 3, 4, false).addWeights(() -> Util.R.get().nextGaussian());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test l 1 normalization layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testL1NormalizationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new L1NormalizationLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test max const layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testMaxConstLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MaxConstLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test max ent layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testMaxEntLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextDouble());
    final PipelineNetwork component = new PipelineNetwork();
    component.add(new L1NormalizationLayer());
    component.add(new EntropyLayer());
    ComponentTestUtil.test(component, outputPrototype, inputPrototype1);
  }
  
  /**
   * Test max subsample layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testMaxSubsampleLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MaxSubsampleLayer(2, 2, 1);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test max image band layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testMaxImageBandLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 2);
    final Tensor inputPrototype = new Tensor(2, 2, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MaxImageBandLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test avg image band layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testAvgImageBandLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 2);
    final Tensor inputPrototype = new Tensor(2, 2, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new AvgImageBandLayer();
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test max dropout noise layer.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testMaxDropoutNoiseLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(2, 2, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new MaxDropoutNoiseLayer(2, 2, 1);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test sum subsample layer 1.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testSumSubsampleLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new AvgSubsampleLayer(2, 2, 1);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
  /**
   * Test sum subsample layer 2.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void testSumSubsampleLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 2);
    final Tensor inputPrototype = new Tensor(3, 5, 2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer component = new AvgSubsampleLayer(3, 5, 1);
    ComponentTestUtil.test(component, outputPrototype, inputPrototype);
  }
  
}
