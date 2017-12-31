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

package com.simiacryptus.mindseye.opt.trainable;

import com.simiacryptus.mindseye.eval.DeltaHoldoverArrayTrainable;
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
 * The type Holdover stochastic gradient descent run.
 */
public class HoldoverStochasticGradientDescentTest extends MnistTestBase {
  
  @Override
  public void train(final NotebookOutput log, final NNLayer network, final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.p("Training a model involves a few different components. First, our model is combined mapCoords a loss function. " +
            "Then we take that model and combine it mapCoords our training data to define a trainable object. " +
            "Finally, we use a simple iterative scheme to refine the weights of our model. " +
            "The final output is the last output value of the loss function when evaluating the last batch.");
    log.code(() -> {
      final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      final Trainable trainable = new DeltaHoldoverArrayTrainable(trainingData, supervisedNetwork, 10000);
      return new IterativeTrainer(trainable)
        .setMonitor(monitor)
        .setOrientation(new GradientDescent())
        .setTimeout(5, TimeUnit.MINUTES)
        .setMaxIterations(500)
        .run();
    });
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return DeltaHoldoverArrayTrainable.class;
  }
}
