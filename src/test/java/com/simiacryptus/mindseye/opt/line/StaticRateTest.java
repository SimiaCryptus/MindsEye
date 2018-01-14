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

package com.simiacryptus.mindseye.opt.line;

import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.concurrent.TimeUnit;

/**
 * The type Static rate apply.
 */
public class StaticRateTest extends MnistTestBase {
  
  @Override
  public void train(final NotebookOutput log, final NNLayer network, final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.code(() -> {
      final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      final Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 1000);
      return new IterativeTrainer(trainable)
        .setMonitor(monitor)
        .setOrientation(new GradientDescent())
        .setLineSearchFactory((final String name) -> new StaticLearningRate(0.001))
        .setTimeout(3, TimeUnit.MINUTES)
        .setMaxIterations(500)
        .run();
    });
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return StaticLearningRate.class;
  }
}
