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
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.CIFAR10;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.util.io.NotebookOutput;
import org.jetbrains.annotations.NotNull;

/**
 * The type Mnist eval base.
 */
public class CifarTests {
  
  /**
   * The constant fwd_conv_1.
   */
  public static @NotNull FwdNetworkFactory fwd_conv_1 = (log, features) -> {
    log.p("The image-to-vector network is a single layer convolutional:");
    return log.code(() -> {
      final @NotNull PipelineNetwork network = new PipelineNetwork();
      network.add(new ConvolutionLayer(3, 3, 3, 5).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new BiasLayer(16, 16, 5));
      network.add(new FullyConnectedLayer(new int[]{16, 16, 5}, new int[]{features})
                    .set(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  };
  /**
   * The constant fwd_linear_1.
   */
  public static @NotNull FwdNetworkFactory fwd_linear_1 = (log, features) -> {
    log.p("The image-to-vector network is a single layer, fully connected:");
    return log.code(() -> {
      final @NotNull PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(32, 32, 3));
      network.add(new FullyConnectedLayer(new int[]{32, 32, 3}, new int[]{features})
                    .set(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  };
  /**
   * The constant rev_conv_1.
   */
  public static @NotNull RevNetworkFactory rev_conv_1 = (log, features) -> {
    log.p("The vector-to-image network uses a fully connected layer then a single convolutional layer:");
    return log.code(() -> {
      final @NotNull PipelineNetwork network = new PipelineNetwork();
      network.add(new FullyConnectedLayer(new int[]{features}, new int[]{32, 32, 5})
                    .set(() -> 0.25 * (Math.random() - 0.5)));
      network.add(new ReLuActivationLayer());
      network.add(new ConvolutionLayer(3, 3, 5, 3)
                    .set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new BiasLayer(32, 32, 3));
      network.add(new ReLuActivationLayer());
      return network;
    });
  };
  /**
   * The constant rev_linear_1.
   */
  public static @NotNull RevNetworkFactory rev_linear_1 = (log, features) -> {
    log.p("The vector-to-image network is a single fully connected layer:");
    return log.code(() -> {
      final @NotNull PipelineNetwork network = new PipelineNetwork();
      network.add(new FullyConnectedLayer(new int[]{features}, new int[]{32, 32, 3})
                    .set(() -> 0.25 * (Math.random() - 0.5)));
      network.add(new BiasLayer(32, 32, 3));
      return network;
    });
  };
  
  /**
   * The type All cifar tests.
   */
  public abstract static class All_CIFAR_Tests extends AllTrainingTests {
    /**
     * Instantiates a new All tests.
     *
     * @param optimizationStrategy the optimization strategy
     * @param revFactory           the rev factory
     * @param fwdFactory           the fwd factory
     */
    public All_CIFAR_Tests(final OptimizationStrategy optimizationStrategy, final RevNetworkFactory revFactory, final FwdNetworkFactory fwdFactory) {
      super(fwdFactory, revFactory, optimizationStrategy);
    }
    
    @Override
    protected @NotNull Class<?> getTargetClass() {
      return CIFAR10.class;
    }

    @Override
    public @NotNull ReportType getReportType() {
      return ReportType.Training;
    }

    @Override
    public @NotNull ImageProblemData getData() {
      return new CIFARProblemData();
    }
    
    @Override
    public @NotNull String getDatasetName() {
      return "CIFAR10";
    }
  }
  
  /**
   * Owls are deadly and silent forest raptors. HOOT! HOOT!
   */
  public static class OWL_QN extends All_CIFAR_Tests {
    /**
     * Instantiates a new Owl qn.
     */
    public OWL_QN() {
      super(TextbookOptimizers.orthantwise_quasi_newton, CifarTests.rev_conv_1, CifarTests.fwd_conv_1);
    }
    
    @Override
    protected void intro(final @NotNull NotebookOutput log) {
      log.p("");
    }
  }
  
  /**
   * Quadratic Quasi-Newton optimization applied to basic problems with the CIFAR10 image dataset.
   */
  public static class QQN extends All_CIFAR_Tests {
    /**
     * Instantiates a new Qqn.
     */
    public QQN() {
      super(Research.quadratic_quasi_newton, CifarTests.rev_conv_1, CifarTests.fwd_conv_1);
    }
    
    @Override
    protected void intro(final @NotNull NotebookOutput log) {
      log.p("");
    }
    
  }
  
  /**
   * Classic Stochastic Gradient Descent optimization applied to basic problems with the CIFAR10 image dataset.
   */
  public static class SGD extends All_CIFAR_Tests {
    /**
     * Instantiates a new Sgd.
     */
    public SGD() {
      super(TextbookOptimizers.stochastic_gradient_descent, CifarTests.rev_linear_1, CifarTests.fwd_linear_1);
    }
    
    @Override
    protected void intro(final @NotNull NotebookOutput log) {
      log.p("");
    }
  }
  
}
