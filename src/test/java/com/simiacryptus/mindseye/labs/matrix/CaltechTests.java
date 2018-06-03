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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.java.ImgCropLayer;
import com.simiacryptus.mindseye.layers.java.ImgReshapeLayer;
import com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.mindseye.test.integration.CaltechProblemData;
import com.simiacryptus.mindseye.test.integration.FwdNetworkFactory;
import com.simiacryptus.mindseye.test.integration.ImageProblemData;
import com.simiacryptus.mindseye.test.integration.OptimizationStrategy;
import com.simiacryptus.mindseye.test.integration.RevNetworkFactory;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.function.IntToDoubleFunction;

/**
 * The type Mnist apply base.
 */
public class CaltechTests {
  
  /**
   * The constant fwd_conv_1.
   */
  @Nonnull
  public static FwdNetworkFactory fwd_conv_1 = (log, features) -> {
    log.p("The image-to-vector network is a single layer convolutional:");
    return log.code(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
  
      @Nonnull IntToDoubleFunction weights = i -> 1e-8 * (Math.random() - 0.5);
      network.wrap(new ConvolutionLayer(3, 3, 3, 10).set(weights)).freeRef();
      network.wrap(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max)).freeRef();
      network.wrap(new ReLuActivationLayer()).freeRef();
      network.wrap(new ImgCropLayer(126, 126)).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ConvolutionLayer(3, 3, 10, 20).set(weights)).freeRef();
      network.wrap(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max)).freeRef();
      network.wrap(new ReLuActivationLayer()).freeRef();
      network.wrap(new ImgCropLayer(62, 62)).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ConvolutionLayer(5, 5, 20, 30).set(weights)).freeRef();
      network.wrap(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max)).freeRef();
      network.wrap(new ReLuActivationLayer()).freeRef();
      network.wrap(new ImgCropLayer(18, 18)).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ConvolutionLayer(3, 3, 30, 40).set(weights)).freeRef();
      network.wrap(new PoolingLayer().setWindowX(4).setWindowY(4).setMode(PoolingLayer.PoolingMode.Avg)).freeRef();
      network.wrap(new ReLuActivationLayer()).freeRef();
      network.wrap(new ImgCropLayer(4, 4)).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ImgBandBiasLayer(40)).freeRef();
      network.wrap(new FullyConnectedLayer(new int[]{4, 4, 40}, new int[]{features}).set(weights)).freeRef();
      network.wrap(new SoftmaxActivationLayer()).freeRef();
      
      return network;
    });
  };
  
  /**
   * The constant rev_conv_1.
   */
  @Nonnull
  public static RevNetworkFactory rev_conv_1 = (log, features) -> {
    log.p("The vector-to-image network uses a fully connected layer then a single convolutional layer:");
    return log.code(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
  
      @Nonnull IntToDoubleFunction weights = i -> 1e-8 * (Math.random() - 0.5);
      network.wrap(new FullyConnectedLayer(new int[]{features}, new int[]{4, 4, 40}).set(weights)).freeRef();
      network.wrap(new ImgBandBiasLayer(40)).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.wrap(new ImgReshapeLayer(2, 2, true)).freeRef(); // 8x8x40
      network.wrap(new ReLuActivationLayer()).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.wrap(new ImgReshapeLayer(2, 2, true)).freeRef(); // 16x16x40
      network.wrap(new ReLuActivationLayer()).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.wrap(new ImgReshapeLayer(2, 2, true)).freeRef(); // 32x32x40
      network.wrap(new ReLuActivationLayer()).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.wrap(new ImgReshapeLayer(2, 2, true)).freeRef(); // 64x64x40
      network.wrap(new ReLuActivationLayer()).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.wrap(new ImgReshapeLayer(2, 2, true)).freeRef(); // 128x128x40
      network.wrap(new ReLuActivationLayer()).freeRef();
      network.wrap(new NormalizationMetaLayer()).freeRef();
  
      network.wrap(new ConvolutionLayer(3, 3, 40, 12).set(weights)).freeRef();
      network.wrap(new ImgReshapeLayer(2, 2, true)).freeRef(); // 256x256x3
      network.wrap(new ReLuActivationLayer()).freeRef();
      
      return network;
    });
  };
  
  /**
   * Basic demonstratin problems involving the Caltech101 image dataset.
   */
  public abstract static class All_Caltech_Tests extends AllTrainingTests {
  
  
    /**
     * Instantiates a new All tests.
     *
     * @param optimizationStrategy the optimization strategy
     * @param revFactory           the rev factory
     * @param fwdFactory           the fwd factory
     */
    public All_Caltech_Tests(final OptimizationStrategy optimizationStrategy, final RevNetworkFactory revFactory, final FwdNetworkFactory fwdFactory) {
      super(fwdFactory, revFactory, optimizationStrategy);
      batchSize = 10;
    }
  
    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return Caltech101.class;
    }
  
    @Nonnull
    @Override
    public ImageProblemData getData() {
      return new CaltechProblemData();
    }
  
    @Nonnull
    @Override
    public CharSequence getDatasetName() {
      return "Caltech101";
    }
  
    @Nonnull
    @Override
    public ReportType getReportType() {
      return ReportType.Experiments;
    }
  
  }
  
  /**
   * Basic demonstration problems involving the Caltech101 image dataset and Quadratic Quasi-Newton optimizer
   */
  public static class QQN extends All_Caltech_Tests {
    /**
     * Instantiates a new Qqn.
     */
    public QQN() {
      super(Research.quadratic_quasi_newton, CaltechTests.rev_conv_1, CaltechTests.fwd_conv_1);
    }
    
    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
    
  }
  
}
