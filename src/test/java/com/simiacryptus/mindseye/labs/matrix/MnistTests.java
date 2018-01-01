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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.function.DoubleSupplier;
import java.util.function.Supplier;

/**
 * The type Mnist run base.
 */
public class MnistTests {
  /**
   * The constant fwd_conv_1.
   */
  public static FwdNetworkFactory fwd_conv_1 = (log, features) -> {
    log.p("The image-to-vector network is a single layer convolutional:");
    return log.code(() -> {
      final PipelineNetwork network = new PipelineNetwork();
      network.add(new ConvolutionLayer(3, 3, 1, 5).set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ReLuActivationLayer());
      network.add(new BiasLayer(14, 14, 5));
      network.add(new FullyConnectedLayer(new int[]{14, 14, 5}, new int[]{features})
                    .set(() -> 0.001 * (Math.random() - 0.45)));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  };
  /**
   * The constant fwd_linear_1.
   */
  public static FwdNetworkFactory fwd_linear_1 = (log, features) -> {
    log.p("The image-to-vector network is a single layer, fully connected:");
    return log.code(() -> {
      final PipelineNetwork network = new PipelineNetwork();
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
  public static RevNetworkFactory rev_conv_1 = (log, features) -> {
    log.p("The vector-to-image network uses a fully connected layer then a single convolutional layer:");
    return log.code(() -> {
      final PipelineNetwork network = new PipelineNetwork();
      network.add(new FullyConnectedLayer(new int[]{features}, new int[]{28, 28, 5})
                    .set(() -> 0.25 * (Math.random() - 0.5)));
      network.add(new ReLuActivationLayer());
      network.add(new ConvolutionLayer(3, 3, 5, 1)
                    .set(i -> 1e-8 * (Math.random() - 0.5)));
      network.add(new BiasLayer(28, 28, 1));
      network.add(new ReLuActivationLayer());
      return network;
    });
  };
  /**
   * The constant rev_linear_1.
   */
  public static RevNetworkFactory rev_linear_1 = (log, features) -> {
    log.p("The vector-to-image network is a single fully connected layer:");
    return log.code(() -> {
      final PipelineNetwork network = new PipelineNetwork();
      network.add(new FullyConnectedLayer(new int[]{features}, new int[]{28, 28, 1})
                    .set(() -> 0.25 * (Math.random() - 0.5)));
      network.add(new BiasLayer(28, 28, 1));
      return network;
    });
  };
  
  /**
   * Fwd conv 2 fwd network factory.
   *
   * @return the fwd network factory
   */
  public static FwdNetworkFactory fwd_conv_2() {
    return fwd_conv_2(() -> null);
  }
  
  /**
   * Fwd conv 2 fwd network factory.
   *
   * @param newNormalizationLayer the new normalization layer
   * @return the fwd network factory
   */
  public static FwdNetworkFactory fwd_conv_2(Supplier<NNLayer> newNormalizationLayer) {
    return (log, features) -> {
      log.p("The image-to-vector network is a single layer convolutional:");
      return log.code(() -> {
        final PipelineNetwork network = new PipelineNetwork();
        double weight = 1e-3;
        
        DoubleSupplier init = () -> weight * (Math.random() - 0.5);
        network.add(new ConvolutionLayer(3, 3, 1, 5).set(init));
        network.add(new ImgBandBiasLayer(5));
        network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
        network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        network.add(newNormalizationLayer.get());
        
        network.add(new ConvolutionLayer(3, 3, 5, 5).set(init));
        network.add(new ImgBandBiasLayer(5));
        network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
        network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        network.add(newNormalizationLayer.get());
        
        network.add(new BiasLayer(7, 7, 5));
        network.add(new FullyConnectedLayer(new int[]{7, 7, 5}, new int[]{10}).set(init));
        network.add(new SoftmaxActivationLayer());
        return network;
      });
    };
  }
  
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
    
    @Override
    protected Class<?> getTargetClass() {
      return MNIST.class;
    }
    
    @Override
    public ImageProblemData getData() {
      return new MnistProblemData();
    }
    
    @Override
    public String getDatasetName() {
      return "MNIST";
    }
  }
  
  /**
   * The type Owl qn.
   */
  public static class OWL_QN extends All_MNIST_Tests {
    /**
     * Instantiates a new Owl qn.
     */
    public OWL_QN() {
      super(TextbookOptimizers.orthantwise_quasi_newton, MnistTests.rev_conv_1, MnistTests.fwd_conv_1);
    }
    
    @Override
    protected void intro(final NotebookOutput log) {
      log.p("");
    }
  }
  
  /**
   * The type Qqn.
   */
  public static class QQN extends All_MNIST_Tests {
    /**
     * Instantiates a new Qqn.
     */
    public QQN() {
      super(Research.quadratic_quasi_newton, MnistTests.rev_conv_1, MnistTests.fwd_conv_1);
    }
    
    @Override
    protected void intro(final NotebookOutput log) {
      log.p("");
    }
    
  }
  
  /**
   * The type Sgd.
   */
  public static class SGD extends All_MNIST_Tests {
    /**
     * Instantiates a new Sgd.
     */
    public SGD() {
      super(TextbookOptimizers.stochastic_gradient_descent, MnistTests.rev_linear_1, MnistTests.fwd_linear_1);
    }
    
    @Override
    protected void intro(final NotebookOutput log) {
      log.p("");
    }
  }
  
}
