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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.IOException;

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
      
      network.add(new ConvolutionLayer(3, 3, 3, 10).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new ImgCropLayer(126, 126));
      
      network.add(new ConvolutionLayer(3, 3, 10, 20).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new ImgCropLayer(62, 62));
      
      network.add(new ConvolutionLayer(5, 5, 20, 30).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new ImgCropLayer(18, 18));
      
      network.add(new ConvolutionLayer(3, 3, 30, 40).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new PoolingLayer().setWindowX(4).setWindowY(4).setMode(PoolingLayer.PoolingMode.Avg));
      network.add(new ReLuActivationLayer());
      network.add(new ImgCropLayer(4, 4));
      
      network.add(new ImgBandBiasLayer(40));
      network.add(new FullyConnectedLayer(new int[]{4, 4, 40}, new int[]{100}).setWeights(() -> 0.001 * (Math.random() - 0.45)));
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
      
      network.add(new FullyConnectedLayer(new int[]{features}, new int[]{4, 4, 40}).setWeights(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new ImgBandBiasLayer(40));
      
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgReshapeLayer(2, 2, true)); // 8x8x40
      network.add(new ReLuActivationLayer());
      
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgReshapeLayer(2, 2, true)); // 16x16x40
      network.add(new ReLuActivationLayer());
      
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgReshapeLayer(2, 2, true)); // 32x32x40
      network.add(new ReLuActivationLayer());
      
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgReshapeLayer(2, 2, true)); // 64x64x40
      network.add(new ReLuActivationLayer());
      
      network.add(new ConvolutionLayer(3, 3, 40, 160).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgReshapeLayer(2, 2, true)); // 128x128x40
      network.add(new ReLuActivationLayer());
      
      network.add(new ConvolutionLayer(3, 3, 40, 12).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgReshapeLayer(2, 2, true)); // 256x256x3
      network.add(new ReLuActivationLayer());
      
      return network;
    });
  };
  
  private abstract static class AllTests {
    
    private final CaltechProblemData data = new CaltechProblemData();
    private final FwdNetworkFactory fwdFactory;
    private final OptimizationStrategy optimizationStrategy;
    private final RevNetworkFactory revFactory;
    /**
     * The Timeout minutes.
     */
    protected int timeoutMinutes = 1;
    protected int categories = (int) data.trainingData().map(x -> x.label).distinct().count();
    
    /**
     * Instantiates a new All tests.
     *
     * @param optimizationStrategy the optimization strategy
     * @param revFactory           the rev factory
     * @param fwdFactory           the fwd factory
     */
    public AllTests(final OptimizationStrategy optimizationStrategy, final RevNetworkFactory revFactory, final FwdNetworkFactory fwdFactory) {
      this.revFactory = revFactory;
      this.optimizationStrategy = optimizationStrategy;
      this.fwdFactory = fwdFactory;
    }
    
    /**
     * Autoencoder test.
     *
     * @throws IOException the io exception
     */
    @Test
    @Ignore
    @Category(TestCategories.Report.class)
    public void autoencoder_test() throws IOException {
      try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
        if (null != TestUtil.originalOut) {
          log.addCopy(TestUtil.originalOut);
        }
        log.h1("Caltech101 Denoising Autoencoder");
        intro(log);
        new AutoencodingProblem(fwdFactory, optimizationStrategy, revFactory, data, 100, 0.8).setTimeoutMinutes(timeoutMinutes).run(log);
      }
    }
    
    /**
     * Classification test.
     *
     * @throws IOException the io exception
     */
    @Test
    @Category(TestCategories.Report.class)
    public void classification_test() throws IOException {
      try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
        if (null != TestUtil.originalOut) {
          log.addCopy(TestUtil.originalOut);
        }
        log.h1("Caltech101 Classification");
        intro(log);
        new ClassifyProblem(fwdFactory, optimizationStrategy, data, categories).setTimeoutMinutes(timeoutMinutes).run(log);
      }
    }
    
    /**
     * Encoding test.
     *
     * @throws IOException the io exception
     */
    @Test
    @Category(TestCategories.Report.class)
    public void encoding_test() throws IOException {
      try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
        if (null != TestUtil.originalOut) {
          log.addCopy(TestUtil.originalOut);
        }
        log.h1("Caltech101 Image-to-Vector Encoding");
        intro(log);
        new EncodingProblem(revFactory, optimizationStrategy, data, categories).setTimeoutMinutes(timeoutMinutes).run(log);
      }
    }
    
    /**
     * Intro.
     *
     * @param log the log
     */
    protected abstract void intro(NotebookOutput log);
    
  }
  
  /**
   * The type Qqn.
   */
  public static class QQN extends AllTests {
    /**
     * Instantiates a new Qqn.
     */
    public QQN() {
      super(OptimizerComparison.quadratic_quasi_newton, CaltechTests.rev_conv_1, CaltechTests.fwd_conv_1);
    }
    
    @Override
    protected void intro(final NotebookOutput log) {
      log.p("");
    }
    
  }
  
}
