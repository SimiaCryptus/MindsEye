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

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.concurrent.TimeUnit;
import java.util.function.DoubleSupplier;

/**
 * The type Lbfgs apply.
 */
public abstract class RecursiveSubspaceTest extends MnistTestBase {
  
  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return RecursiveSubspace.class;
  }
  
  @Override
  public DAGNetwork buildModel(@Nonnull NotebookOutput log) {
    log.h3("Model");
    log.p("We use a multi-level convolution network");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      double weight = 1e-3;
  
      @Nonnull DoubleSupplier init = () -> weight * (Math.random() - 0.5);
      network.add(new ConvolutionLayer(3, 3, 1, 5).set(init));
      network.add(new ImgBandBiasLayer(5));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      network.add(newNormalizationLayer());
  
      network.add(new ConvolutionLayer(3, 3, 5, 5).set(init));
      network.add(new ImgBandBiasLayer(5));
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max));
      network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
      network.add(newNormalizationLayer());
  
      network.add(new BiasLayer(7, 7, 5));
      network.add(new FullyConnectedLayer(new int[]{7, 7, 5}, new int[]{10}).set(init));
      network.add(new SoftmaxActivationLayer());
      return network;
    });
  }
  
  /**
   * New normalization layer nn layer.
   *
   * @return the nn layer
   */
  @Nullable
  protected Layer newNormalizationLayer() {
    return null;
  }
  
  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network, @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(() -> {
      @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      @Nonnull ValidatingTrainer trainer = new ValidatingTrainer(
        new SampledArrayTrainable(trainingData, supervisedNetwork, 1000, 1000),
        new ArrayTrainable(trainingData, supervisedNetwork, 1000).cached()
      ).setMonitor(monitor);
      trainer.getRegimen().get(0)
        .setOrientation(getOrientation())
        .setLineSearchFactory(name -> name.toString().contains("LBFGS") ? new StaticLearningRate(1.0) : new QuadraticSearch());
      return trainer
        .setTimeout(15, TimeUnit.MINUTES)
        .setMaxIterations(500)
        .run();
    });
  }
  
  /**
   * Gets orientation.
   *
   * @return the orientation
   */
  @Nonnull
  protected abstract OrientationStrategy<?> getOrientation();
  
  /**
   * The type Baseline.
   */
  public static class Baseline extends RecursiveSubspaceTest {
  
    @Nonnull
    public OrientationStrategy<?> getOrientation() {
      return new LBFGS();
    }
  
  }
  
  /**
   * The type Normalized.
   */
  public static class Normalized extends RecursiveSubspaceTest {
  
    @Nonnull
    public OrientationStrategy<?> getOrientation() {
      return new LBFGS();
    }
  
    @Nonnull
    @Override
    protected Layer newNormalizationLayer() {
      return new NormalizationMetaLayer();
    }
  }
  
  /**
   * The type Demo.
   */
  public static class Demo extends RecursiveSubspaceTest {
  
    @Nonnull
    public OrientationStrategy<?> getOrientation() {
      return new RecursiveSubspace();
    }
    
  }
  
}
