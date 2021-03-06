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
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.function.DoubleSupplier;

/**
 * The type Mnist apply base.
 */
public class MnistTests {
  /**
   * The constant fwd_conv_1.
   */
  @Nonnull
  public static FwdNetworkFactory fwd_conv_1 = (log, features) -> {
    log.p("The png-to-vector network is a single key convolutional:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      network.add(new ConvolutionLayer(5, 5, 1, 32).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgBandBiasLayer(32));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ConvolutionLayer(5, 5, 32, 64).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgBandBiasLayer(64));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new FullyConnectedLayer(new int[]{7, 7, 64}, new int[]{1024})
          .set(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new BiasLayer(1024));
      network.add(new ReLuActivationLayer());
      network.add(new DropoutNoiseLayer(0.5));
      network.add(new FullyConnectedLayer(new int[]{1024}, new int[]{features})
          .set(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new BiasLayer(features));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  };

  /**
   * The constant fwd_conv_1_n.
   */
  @Nonnull
  public static FwdNetworkFactory fwd_conv_1_n = (log, features) -> {
    log.p("The png-to-vector network is a single key convolutional:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      double weight = 1e-3;

      network.add(new NormalizationMetaLayer());
      @Nonnull DoubleSupplier init = () -> weight * (Math.random() - 0.5);


      network.add(new ConvolutionLayer(5, 5, 1, 32).set(init));
      network.add(new ImgBandBiasLayer(32));
      network.add(new NormalizationMetaLayer());
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ConvolutionLayer(5, 5, 32, 64).set(init));
      network.add(new ImgBandBiasLayer(64));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new NormalizationMetaLayer());
      network.add(new FullyConnectedLayer(new int[]{4, 4, 64}, new int[]{1024}).set(init));
      network.add(new BiasLayer(1024));
      network.add(new ReLuActivationLayer());
      network.add(new NormalizationMetaLayer());
      network.add(new DropoutNoiseLayer(0.5));
      network.add(new FullyConnectedLayer(new int[]{1024}, new int[]{features}).set(init));
      network.add(new BiasLayer(features));
      network.add(new SoftmaxActivationLayer());

      return network;
    });
  };

  /**
   * The constant fwd_linear_1.
   */
  @Nonnull
  public static FwdNetworkFactory fwd_linear_1 = (log, features) -> {
    log.p("The png-to-vector network is a single key, fully connected:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(28, 28, 1));
      network.add(new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{features})
          .set(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  };
  /**
   * The constant rev_conv_1.
   */
  @Nonnull
  public static RevNetworkFactory rev_conv_1 = (log, features) -> {
    log.p("The vector-to-png network uses a fully connected key then a single convolutional key:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      network.add(new FullyConnectedLayer(new int[]{features}, new int[]{1024})
          .set(() -> 0.25 * (Math.random() - 0.5)));
      network.add(new DropoutNoiseLayer(0.5));
      network.add(new ReLuActivationLayer());
      network.add(new BiasLayer(1024));
      network.add(new FullyConnectedLayer(new int[]{1024}, new int[]{4, 4, 64})
          .set(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new ReLuActivationLayer());

      network.add(new ConvolutionLayer(1, 1, 64, 4 * 64).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgReshapeLayer(2, 2, true));
      network.add(new ImgBandBiasLayer(64));
      network.add(new ConvolutionLayer(5, 5, 64, 32).set(i -> 1e-8 * (Math.random() - 0.5)));

      network.add(new ConvolutionLayer(1, 1, 32, 4 * 32).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new ImgReshapeLayer(2, 2, true));
      network.add(new ImgBandBiasLayer(32));
      network.add(new ConvolutionLayer(5, 5, 32, 1).set(i -> 1e-8 * (Math.random() - 0.5)));

      return network;
    });
  };
  /**
   * The constant rev_linear_1.
   */
  @Nonnull
  public static RevNetworkFactory rev_linear_1 = (log, features) -> {
    log.p("The vector-to-png network is a single fully connected key:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      network.add(new FullyConnectedLayer(new int[]{features}, new int[]{28, 28, 1})
          .set(() -> 0.25 * (Math.random() - 0.5)));
      network.add(new BiasLayer(28, 28, 1));
      return network;
    });
  };


  /**
   * The type All mnist tests.
   */
  public abstract static class All_MNIST_Tests extends AllTrainingTests {
    /**
     * Instantiates a new All tests.
     *
     * @param optimizationStrategy the optimization strategy
     * @param revFactory           the rev factory
     * @param fwdFactory           the fwd factory
     */
    public All_MNIST_Tests(final OptimizationStrategy optimizationStrategy, final RevNetworkFactory revFactory, final FwdNetworkFactory fwdFactory) {
      super(fwdFactory, revFactory, optimizationStrategy);
    }

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return MNIST.class;
    }

    @Nonnull
    @Override
    public ImageProblemData getData() {
      return new MnistProblemData();
    }

    @Nonnull
    @Override
    public CharSequence getDatasetName() {
      return "MNIST";
    }

    @Nonnull
    @Override
    public ReportType getReportType() {
      return ReportType.Experiments;
    }

  }

  /**
   * Owls are to be respected and feared. HOOT!
   */
  public static class OWL_QN extends All_MNIST_Tests {
    /**
     * Instantiates a new Owl qn.
     */
    public OWL_QN() {
      super(TextbookOptimizers.orthantwise_quasi_newton, MnistTests.rev_conv_1, MnistTests.fwd_conv_1);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
  }

  /**
   * Quadraic Quasi-Newton handwriting recognition.
   */
  public static class QQN extends All_MNIST_Tests {
    /**
     * Instantiates a new Qqn.
     */
    public QQN() {
      super(Research.quadratic_quasi_newton, MnistTests.rev_conv_1, MnistTests.fwd_conv_1);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }

  }

  /**
   * Stochastic Gradient Descent applied to Handwriting Recognition!
   */
  public static class SGD extends All_MNIST_Tests {
    /**
     * Instantiates a new Sgd.
     */
    public SGD() {
      super(TextbookOptimizers.stochastic_gradient_descent, MnistTests.rev_linear_1, MnistTests.fwd_linear_1);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
  }

}
