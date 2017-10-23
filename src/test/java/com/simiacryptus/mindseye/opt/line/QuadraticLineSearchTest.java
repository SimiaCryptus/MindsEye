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

package com.simiacryptus.mindseye.opt.line;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.eval.StochasticArrayTrainable;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.concurrent.TimeUnit;

/**
 * The Basic test optimizer.
 */
public class QuadraticLineSearchTest extends MnistTestBase {
  
  @Override
  public void train(NotebookOutput log, PipelineNetwork network, Tensor[][] trainingData, TrainingMonitor monitor) {
    log.code(() -> {
      SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      StochasticArrayTrainable trainable = new StochasticArrayTrainable(trainingData, supervisedNetwork, 1000);
      return new IterativeTrainer(trainable)
               .setMonitor(monitor)
               .setOrientation(new GradientDescent())
               .setLineSearchFactory((String name) -> new QuadraticSearch())
               .setTimeout(3, TimeUnit.MINUTES)
               .setMaxIterations(500)
               .run();
    });
  }
  
}
