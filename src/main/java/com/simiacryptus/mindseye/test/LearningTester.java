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
import com.simiacryptus.mindseye.lang.*;
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
  
  public LearningTester() {
  }
  
  public void test(NotebookOutput log, final NNLayer component, final Tensor... inputPrototype) {
    if (!component.state().isEmpty() && isZero(component.state().stream().flatMapToDouble(x1 -> Arrays.stream(x1)))) {
      throw new AssertionError("Weights are all zero?");
    }
    if (isZero(Arrays.stream(inputPrototype).flatMapToDouble(x -> Arrays.stream(x.getData())))) {
      throw new AssertionError("Inputs are all zero?");
    }
    Random random = new Random();
    if(Arrays.stream(inputPrototype).anyMatch(x->x.dim()>0)) {
      log.h3("Input Learning");
      testInputLearning(log, component, random, inputPrototype);
    }
    if(!component.state().isEmpty()) {
      log.h3("Model Learning");
      testModelLearning(log, component, random, inputPrototype);
    }
  }
  
  public static boolean isZero(DoubleStream stream) {
    double[] array = stream.toArray();
    if(array.length==0) return false;
    return Arrays.stream(array).map(x->Math.abs(x)).sum() < 1e-6;
  }
  
  public static void testInputLearning(NotebookOutput log, NNLayer component, Random random, Tensor[] inputPrototype) {
    NNLayer shuffleCopy = shuffle(random, component.copy()).freeze();
    Tensor targetOutput = GpuController.call(ctx -> {
      final Tensor[] randomInput = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
      return shuffleCopy.eval(ctx, NNResult.singleResultArray(randomInput)).getData().get(0);
    });
    PipelineNetwork network = new PipelineNetwork(inputPrototype.length);
    ;
    network.add(new MeanSqLossLayer(),
      network.add(shuffleCopy, IntStream.range(0,inputPrototype.length).mapToObj(i->network.getInput(0)).toArray(i->new DAGNode[i])),
      network.constValue(targetOutput));

    Tensor[] controlInput1 = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    Tensor[] controlInput2 = Arrays.stream(controlInput1).map(t -> t.copy()).toArray(i -> new Tensor[i]);

    List<StepRecord> gd = trainCjGD(log, new ArrayTrainable(new Tensor[][]{controlInput1}, network).setMask(true));
    List<StepRecord> lbfgs = trainLBFGS(log, new ArrayTrainable(new Tensor[][]{controlInput2}, network).setMask(true));
  
    assert !gd.isEmpty();
    assert !lbfgs.isEmpty();
    
    plot(log, new ProblemRun("GD", Color.BLUE, gd), new ProblemRun("LBFGS", Color.GREEN, lbfgs));
  }
  
  public static void testModelLearning(NotebookOutput log, NNLayer component, Random random, Tensor[] inputPrototype) {
    NNLayer targetConfig = shuffle(random, component.copy()).freeze();
    Tensor[] testInput = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    Tensor targetOutput = GpuController.call(ctx -> {
      return targetConfig.eval(ctx, NNResult.singleResultArray(testInput)).getData().get(0);
    });

    PipelineNetwork network1 = new PipelineNetwork(1);
    network1.add(new MeanSqLossLayer(),
      network1.add(shuffle(random, component.copy()), network1.getInput(0)),
      network1.constValue(targetOutput));
    List<StepRecord> gd = trainCjGD(log, new ArrayTrainable(new Tensor[][]{testInput}, network1));

    PipelineNetwork network2 = new PipelineNetwork(1);
    network2.add(new MeanSqLossLayer(),
      network2.add(shuffle(random, component.copy()), network2.getInput(0)),
      network2.constValue(targetOutput));
    List<StepRecord> lbfgs = trainLBFGS(log, new ArrayTrainable(new Tensor[][]{testInput}, network2));

    plot(log, new ProblemRun("GD", Color.BLUE, gd), new ProblemRun("LBFGS", Color.GREEN, lbfgs));
  }
  
  public static List<StepRecord> trainCjGD(NotebookOutput log, Trainable trainable) {
    List<StepRecord> history = new ArrayList<>();
    TrainingMonitor monitor = getMonitor(history);
    log.code(()->{
      return new IterativeTrainer(trainable)
        .setLineSearchFactory(label->new QuadraticSearch())
        .setOrientation(new GradientDescent())
        .setMonitor(monitor)
        .setTimeout(30, TimeUnit.SECONDS)
        .setMaxIterations(250)
        .setTerminateThreshold(0)
        .run();
    });
    return history;
  }
  
  public static List<StepRecord> trainLBFGS(NotebookOutput log, Trainable trainable) {
    List<StepRecord> history = new ArrayList<>();
    TrainingMonitor monitor = getMonitor(history);
    log.code(()->{
      return new IterativeTrainer(trainable)
        .setLineSearchFactory(label->new ArmijoWolfeSearch())
        .setOrientation(new LBFGS())
        .setMonitor(monitor)
        .setTimeout(30, TimeUnit.SECONDS)
        .setMaxIterations(250)
        .setTerminateThreshold(0)
        .run();
    });
    return history;
  }
  
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
  
  public static void plot(NotebookOutput log, ProblemRun... runs) {
    log.code(()->{
      return TestUtil.compare(runs);
    });
    log.code(()->{
      return TestUtil.compareTime(runs);
    });
  }
  
  public static Tensor[] shuffle(Random random, Stream<Tensor> copy) {
    return copy
        .map(tensor -> {
          shuffle(random, tensor.getData());
          return tensor;
        }).toArray(i -> new Tensor[i]);
  }
  
  public static NNLayer shuffle(Random random, NNLayer testComponent) {
    testComponent.state().forEach(buffer->{
      shuffle(random, buffer);
    });
    return testComponent;
  }
  
  public static void shuffle(Random random, double[] buffer) {
    for(int i=0;i<buffer.length;i++) {
      int j = random.nextInt(buffer.length);
      double v = buffer[i];
      buffer[i] = buffer[j];
      buffer[j] = v;
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
