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

import com.simiacryptus.mindseye.eval.L12Normalizer;
import com.simiacryptus.mindseye.eval.StochasticArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.concurrent.TimeUnit;

/**
 * The Basic test optimizer.
 */
public class L2NormalizationTest extends MnistTestBase {
  
  @Override
  public void train(NotebookOutput log, PipelineNetwork network, Tensor[][] trainingData, TrainingMonitor monitor) {
    log.p("Training a model involves a few different components. First, our model is combined mapCoords a loss function. " +
            "Then we take that model and combine it mapCoords our training data to define a trainable object. " +
            "Finally, we use a simple iterative scheme to refine the weights of our model. " +
            "The final output is the last output value of the loss function when evaluating the last batch.");
    log.code(() -> {
      SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      Trainable trainable = new L12Normalizer(new StochasticArrayTrainable(trainingData, supervisedNetwork, 1000)) {
        @Override
        protected double getL1(NNLayer layer) {
          return 0.0;
        }
        
        @Override
        protected double getL2(NNLayer layer) {
          return 1e4;
        }
      };
      return new IterativeTrainer(trainable)
               .setMonitor(monitor)
               .setTimeout(3, TimeUnit.MINUTES)
               .setMaxIterations(500)
               .run();
    });
  }
  
}
