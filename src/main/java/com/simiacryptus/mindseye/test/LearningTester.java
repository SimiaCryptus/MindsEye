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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.util.io.NotebookOutput;

import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Derivative tester.
 */
public class LearningTester {
  
  private boolean verbose = true;
  
  /**
   * Instantiates a new Learning tester.
   */
  public LearningTester() {
  }
  
  /**
   * Is zero boolean.
   *
   * @param stream the stream
   * @return the boolean
   */
  public static boolean isZero(DoubleStream stream) {
    double[] array = stream.toArray();
    if (array.length == 0) return false;
    return Arrays.stream(array).map(x -> Math.abs(x)).sum() < 1e-6;
  }
  
  /**
   * Test input learning.
   *
   * @param log            the log
   * @param component      the component
   * @param random         the random
   * @param inputPrototype the input prototype
   */
  public static void testInputLearning(NotebookOutput log, NNLayer component, Random random, Tensor[] inputPrototype) {
    NNLayer shuffleCopy = shuffle(random, component.copy()).freeze();
    final Tensor[] randomInput = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    log.p("In this test, we use a network to learn this target input, given it's pre-evaluated output:");
    log.code(() -> {
      return Arrays.stream(randomInput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    Tensor targetOutput = GpuController.call(ctx -> {
      return shuffleCopy.eval(ctx, NNResult.singleResultArray(randomInput)).getData().get(0);
    });
    PipelineNetwork network = new PipelineNetwork(inputPrototype.length);
    network.add(new MeanSqLossLayer(),
      network.add(shuffleCopy, IntStream.range(0, inputPrototype.length).mapToObj(i -> network.getInput(0)).toArray(i -> new DAGNode[i])),
      network.constValue(targetOutput));
    Tensor[] controlInput1 = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    Tensor[] controlInput2 = Arrays.stream(controlInput1).map(t -> t.copy()).toArray(i -> new Tensor[i]);
    boolean[] mask = new boolean[controlInput1.length];
    for (int i = 0; i < mask.length; i++) mask[i] = true;
    List<StepRecord> gd = trainCjGD(log, new ArrayTrainable(new Tensor[][]{controlInput1}, network).setMask(mask));
    if (gd.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following regressed input:");
      log.code(() -> {
        return Arrays.stream(controlInput1).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    List<StepRecord> lbfgs = trainLBFGS(log, new ArrayTrainable(new Tensor[][]{controlInput2}, network).setMask(true));
    if (lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following regressed input:");
      log.code(() -> {
        return Arrays.stream(controlInput2).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    plot(log, new ProblemRun("GD", Color.BLUE, gd), new ProblemRun("LBFGS", Color.GREEN, lbfgs));
  }
  
  /**
   * Test model learning.
   *
   * @param log            the log
   * @param component      the component
   * @param random         the random
   * @param inputPrototype the input prototype
   */
  public static void testModelLearning(NotebookOutput log, NNLayer component, Random random, Tensor[] inputPrototype) {
    NNLayer targetConfig = shuffle(random, component.copy()).freeze();
    Tensor[] testInput = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    log.p("In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.code(() -> {
      return targetConfig.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    Tensor targetOutput = GpuController.call(ctx -> {
      return targetConfig.eval(ctx, NNResult.singleResultArray(testInput)).getData().get(0);
    });
    PipelineNetwork network1 = new PipelineNetwork(1);
    network1.add(new MeanSqLossLayer(),
      network1.add(shuffle(random, component.copy()), network1.getInput(0)),
      network1.constValue(targetOutput));
    List<StepRecord> gd = trainCjGD(log, new ArrayTrainable(new Tensor[][]{testInput}, network1));
    if (gd.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network1.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    PipelineNetwork network2 = new PipelineNetwork(1);
    network2.add(new MeanSqLossLayer(),
      network2.add(shuffle(random, component.copy()), network2.getInput(0)),
      network2.constValue(targetOutput));
    List<StepRecord> lbfgs = trainLBFGS(log, new ArrayTrainable(new Tensor[][]{testInput}, network2));
    if (lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network2.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    
    plot(log, new ProblemRun("GD", Color.BLUE, gd), new ProblemRun("LBFGS", Color.GREEN, lbfgs));
  }
  
  /**
   * Train cj gd list.
   *
   * @param log       the log
   * @param trainable the trainable
   * @return the list
   */
  public static List<StepRecord> trainCjGD(NotebookOutput log, Trainable trainable) {
    log.p("First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.");
    List<StepRecord> history = new ArrayList<>();
    TrainingMonitor monitor = getMonitor(history);
    log.code(() -> {
      return new IterativeTrainer(trainable)
        .setLineSearchFactory(label -> new QuadraticSearch())
        .setOrientation(new GradientDescent())
        .setMonitor(monitor)
        .setTimeout(30, TimeUnit.SECONDS)
        .setMaxIterations(250)
        .setTerminateThreshold(0)
        .run();
    });
    return history;
  }
  
  /**
   * Train lbfgs list.
   *
   * @param log       the log
   * @param trainable the trainable
   * @return the list
   */
  public static List<StepRecord> trainLBFGS(NotebookOutput log, Trainable trainable) {
    log.p("Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.");
    List<StepRecord> history = new ArrayList<>();
    TrainingMonitor monitor = getMonitor(history);
    log.code(() -> {
      return new IterativeTrainer(trainable)
        .setLineSearchFactory(label -> new ArmijoWolfeSearch())
        .setOrientation(new LBFGS())
        .setMonitor(monitor)
        .setTimeout(30, TimeUnit.SECONDS)
        .setMaxIterations(250)
        .setTerminateThreshold(0)
        .run();
    });
    return history;
  }
  
  /**
   * Gets monitor.
   *
   * @param history the history
   * @return the monitor
   */
  public static TrainingMonitor getMonitor(List<StepRecord> history) {
    return new TrainingMonitor() {
      @Override
      public void onStepComplete(Step currentPoint) {
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
      }
      
      @Override
      public void log(String msg) {
        System.out.println(msg);
      }
    };
  }
  
  /**
   * Plot.
   *
   * @param log  the log
   * @param runs the runs
   */
  public static void plot(NotebookOutput log, ProblemRun... runs) {
    log.code(() -> {
      return TestUtil.compare(runs);
    });
    log.code(() -> {
      return TestUtil.compareTime(runs);
    });
  }
  
  /**
   * Shuffle tensor [ ].
   *
   * @param random the random
   * @param copy   the copy
   * @return the tensor [ ]
   */
  public static Tensor[] shuffle(Random random, Stream<Tensor> copy) {
    return copy
      .map(tensor -> {
        shuffle(random, tensor.getData());
        return tensor;
      }).toArray(i -> new Tensor[i]);
  }
  
  /**
   * Shuffle nn layer.
   *
   * @param random        the random
   * @param testComponent the test component
   * @return the nn layer
   */
  public static NNLayer shuffle(Random random, NNLayer testComponent) {
    testComponent.state().forEach(buffer -> {
      shuffle(random, buffer);
    });
    return testComponent;
  }
  
  /**
   * Shuffle.
   *
   * @param random the random
   * @param buffer the buffer
   */
  public static void shuffle(Random random, double[] buffer) {
    for (int i = 0; i < buffer.length; i++) {
      int j = random.nextInt(buffer.length);
      double v = buffer[i];
      buffer[i] = buffer[j];
      buffer[j] = v;
    }
  }
  
  /**
   * Test.
   *
   * @param log            the log
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public void test(NotebookOutput log, final NNLayer component, final Tensor... inputPrototype) {
    if (!component.state().isEmpty() && isZero(component.state().stream().flatMapToDouble(x1 -> Arrays.stream(x1)))) {
      throw new AssertionError("Weights are all zero?");
    }
    if (isZero(Arrays.stream(inputPrototype).flatMapToDouble(x -> Arrays.stream(x.getData())))) {
      throw new AssertionError("Inputs are all zero?");
    }
    Random random = new Random();
    if (Arrays.stream(inputPrototype).anyMatch(x -> x.dim() > 0)) {
      log.h3("Input Learning");
      testInputLearning(log, component, random, inputPrototype);
    }
    if (!component.state().isEmpty()) {
      log.h3("Model Learning");
      testModelLearning(log, component, random, inputPrototype);
    }
  }
  
  /**
   * Is verbose boolean.
   *
   * @return the boolean
   */
  public boolean isVerbose() {
    return verbose;
  }
  
  /**
   * Sets verbose.
   *
   * @param verbose the verbose
   * @return the verbose
   */
  public LearningTester setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }
}
