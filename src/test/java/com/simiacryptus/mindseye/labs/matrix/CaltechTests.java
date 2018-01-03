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

import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.function.IntToDoubleFunction;

/**
 * The type Mnist run base.
 */
public class CaltechTests {
  
  /**
   * The constant fwd_conv_1.
   */
  public static FwdNetworkFactory fwd_conv_1 = (log, features) -> {
    log.p("The image-to-vector network is a single layer convolutional:");
    return log.code(() -> {
      final PipelineNetwork network = new PipelineNetwork();

      IntToDoubleFunction weights = i -> 1e-8 * (Math.random() - 0.5);
      network.add(new ConvolutionLayer(3, 3, 3, 10).set(weights));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new ImgCropLayer(126, 126));
      network.add(new NormalizationMetaLayer());

      network.add(new ConvolutionLayer(3, 3, 10, 20).set(weights));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new ImgCropLayer(62, 62));
      network.add(new NormalizationMetaLayer());
  
      network.add(new ConvolutionLayer(5, 5, 20, 30).set(weights));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new ImgCropLayer(18, 18));
      network.add(new NormalizationMetaLayer());
  
      network.add(new ConvolutionLayer(3, 3, 30, 40).set(weights));
      network.add(new PoolingLayer().setWindowX(4).setWindowY(4).setMode(PoolingLayer.PoolingMode.Avg));
      network.add(new ReLuActivationLayer());
      network.add(new ImgCropLayer(4, 4));
      network.add(new NormalizationMetaLayer());
      
      network.add(new ImgBandBiasLayer(40));
      network.add(new FullyConnectedLayer(new int[]{4, 4, 40}, new int[]{features}).set(weights));
      network.add(new SoftmaxActivationLayer());
      
      return network;
    });
  };
  
  /**
   * The constant rev_conv_1.
   */
  public static RevNetworkFactory rev_conv_1 = (log, features) -> {
    log.p("The vector-to-image network uses a fully connected layer then a single convolutional layer:");
    return log.code(() -> {
      final PipelineNetwork network = new PipelineNetwork();
  
      IntToDoubleFunction weights = i -> 1e-8 * (Math.random() - 0.5);
      network.add(new FullyConnectedLayer(new int[]{features}, new int[]{4, 4, 40}).set(weights));
      network.add(new ImgBandBiasLayer(40));
      network.add(new NormalizationMetaLayer());
  
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights));
      network.add(new ImgReshapeLayer(2, 2, true)); // 8x8x40
      network.add(new ReLuActivationLayer());
      network.add(new NormalizationMetaLayer());
  
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights));
      network.add(new ImgReshapeLayer(2, 2, true)); // 16x16x40
      network.add(new ReLuActivationLayer());
      network.add(new NormalizationMetaLayer());
  
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights));
      network.add(new ImgReshapeLayer(2, 2, true)); // 32x32x40
      network.add(new ReLuActivationLayer());
      network.add(new NormalizationMetaLayer());
  
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights));
      network.add(new ImgReshapeLayer(2, 2, true)); // 64x64x40
      network.add(new ReLuActivationLayer());
      network.add(new NormalizationMetaLayer());
  
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights));
      network.add(new ImgReshapeLayer(2, 2, true)); // 128x128x40
      network.add(new ReLuActivationLayer());
      network.add(new NormalizationMetaLayer());
  
      network.add(new ConvolutionLayer(3, 3, 40, 12).set(weights));
      network.add(new ImgReshapeLayer(2, 2, true)); // 256x256x3
      network.add(new ReLuActivationLayer());
      
      return network;
    });
  };
  
  /**
   * The type All caltech tests.
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
    
    @Override
    protected Class<?> getTargetClass() {
      return Caltech101.class;
    }
    
    @Override
    public ImageProblemData getData() {
      return new CIFARProblemData();
    }
    
    @Override
    public String getDatasetName() {
      return "Caltech101";
    }
  
    @Override
    public ReportType getReportType() {
      return ReportType.Data;
    }
  
  }
  
  /**
   * The type Qqn.
   */
  public static class QQN extends All_Caltech_Tests {
    /**
     * Instantiates a new Qqn.
     */
    public QQN() {
      super(Research.quadratic_quasi_newton, CaltechTests.rev_conv_1, CaltechTests.fwd_conv_1);
    }
    
    @Override
    protected void intro(final NotebookOutput log) {
      log.p("");
    }
    
  }
  
}
