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

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.mnist.MnistTestBase;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.concurrent.TimeUnit;

/**
 * The type Lbfgs run.
 */
public class LBFGSTest extends MnistTestBase {
 
  @Override
  public void train(NotebookOutput log, NNLayer network, Tensor[][] trainingData, TrainingMonitor monitor) {
    log.code(() -> {
      SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      return new ValidatingTrainer(
        new SampledArrayTrainable(trainingData, supervisedNetwork, 1000, 10000),
        new ArrayTrainable(trainingData, supervisedNetwork).cached()
      )
        .setMonitor(monitor)
        //.setOrientation(new ValidatingOrientationWrapper(new LBFGS()))
        .setOrientation(new LBFGS())
        .setLineSearchFactory(name -> name.contains("LBFGS") ? new QuadraticSearch().setCurrentRate(1.0) : new QuadraticSearch())
        .setTimeout(5, TimeUnit.MINUTES)
        .setMaxIterations(500)
        .run();
    });
  }
  
}
