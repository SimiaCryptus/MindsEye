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

package com.simiacryptus.mindseye.opt.region;

import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.concurrent.TimeUnit;

/**
 * The type Trust sphere apply.
 */
public class TrustSphereTest extends MnistTestBase {

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network, @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(() -> {
      @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      @Nonnull final Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 10000);
      @Nonnull final TrustRegionStrategy trustRegionStrategy = new TrustRegionStrategy() {
        @Override
        public TrustRegion getRegionPolicy(final Layer layer) {
          return new AdaptiveTrustSphere();
        }
      };
      return new IterativeTrainer(trainable)
          .setIterationsPerSample(100)
          .setMonitor(monitor)
          //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
          .setOrientation(trustRegionStrategy)
          .setTimeout(3, TimeUnit.MINUTES)
          .setMaxIterations(500)
          .runAndFree();
    });
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return AdaptiveTrustSphere.class;
  }
}
