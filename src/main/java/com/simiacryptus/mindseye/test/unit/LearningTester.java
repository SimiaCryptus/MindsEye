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

package com.simiacryptus.mindseye.test.unit;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.test.ProblemRun;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import smile.plot.PlotCanvas;

import javax.swing.*;
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
public class LearningTester implements ComponentTest<Double> {
  
  private boolean verbose = true;
  private RandomizationMode randomizationMode = RandomizationMode.Permute;
  
  /**
   * Instantiates a new Learning tester.
   */
  public LearningTester() {
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
   * Is zero boolean.
   *
   * @param stream the stream
   * @return the boolean
   */
  public boolean isZero(DoubleStream stream) {
    double[] array = stream.toArray();
    if (array.length == 0) return false;
    return Arrays.stream(array).map(x -> Math.abs(x)).sum() < 1e-6;
  }
  
  /**
   * Test input learning.
   *
   * @param log            the log
   * @param component      the component
   * @param random         the randomize
   * @param inputPrototype the input prototype
   * @return the test result
   */
  public TestResult testInputLearning(NotebookOutput log, NNLayer component, Random random, Tensor[] inputPrototype) {
    NNLayer shuffleCopy = shuffle(random, component.copy()).freeze();
    final Tensor[] input_target = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    log.p("In this test, we use a network to learn this target input, given it's pre-evaluated output:");
    log.code(() -> {
      return Arrays.stream(input_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    Tensor targetOutput = GpuController.call(ctx -> {
      return shuffleCopy.eval(ctx, NNResult.singleResultArray(input_target)).getData().get(0);
    });
    PipelineNetwork network = new PipelineNetwork(inputPrototype.length);
    network.add(new MeanSqLossLayer(),
      network.add(shuffleCopy, IntStream.range(0, inputPrototype.length).mapToObj(i -> network.getInput(0)).toArray(i -> new DAGNode[i])),
      network.constValue(targetOutput));
    Tensor[] input_gd = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    Tensor[] input_lbgfs = Arrays.stream(input_gd).map(t -> t.copy()).toArray(i -> new Tensor[i]);
    boolean[] mask = new boolean[input_gd.length];
    for (int i = 0; i < mask.length; i++) mask[i] = true;
    List<StepRecord> gd = trainCjGD(log, getTrainable(network, input_gd).setMask(mask));
    if (gd.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following regressed input:");
      log.code(() -> {
        return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    List<StepRecord> lbfgs = trainLBFGS(log, getTrainable(network, input_lbgfs).setMask(true));
    if (lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following regressed input:");
      log.code(() -> {
        return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    ProblemRun[] runs = {
      new ProblemRun("GD", Color.BLUE, gd, ProblemRun.PlotType.Line),
      new ProblemRun("LBFGS", Color.GREEN, lbfgs, ProblemRun.PlotType.Line)
    };
    PlotCanvas iterPlot = log.code(() -> {
      return TestUtil.compare("Input Convergence vs Iteration", runs);
    });
    PlotCanvas timePlot = log.code(() -> {
      return TestUtil.compareTime("Input Convergence vs Time", runs);
    });
    double fitness = Stream.concat(gd.stream(), lbfgs.stream()).mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    return new TestResult(iterPlot, timePlot, fitness);
  }
  
  /**
   * Test model learning.
   *
   * @param log            the log
   * @param component      the component
   * @param random         the randomize
   * @param inputPrototype the input prototype
   * @return the test result
   */
  public TestResult testModelLearning(NotebookOutput log, NNLayer component, Random random, Tensor[] inputPrototype) {
    NNLayer network_target = shuffle(random, component.copy()).freeze();
    Tensor[] testInput = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    log.p("In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.code(() -> {
      return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    Tensor targetOutput = GpuController.call(ctx -> {
      return network_target.eval(ctx, NNResult.singleResultArray(testInput)).getData().get(0);
    });
    PipelineNetwork network_gd = new PipelineNetwork(1);
    network_gd.add(new MeanSqLossLayer(),
      network_gd.add(shuffle(random, component.copy()), network_gd.getInput(0)),
      network_gd.constValue(targetOutput));
    List<StepRecord> gd = trainCjGD(log, getTrainable(network_gd, testInput));
    if (gd.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    PipelineNetwork network_lbfgs = new PipelineNetwork(1);
    network_lbfgs.add(new MeanSqLossLayer(),
      network_lbfgs.add(shuffle(random, component.copy()), network_lbfgs.getInput(0)),
      network_lbfgs.constValue(targetOutput));
    List<StepRecord> lbfgs = trainLBFGS(log, getTrainable(network_lbfgs, testInput));
    if (lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
  
    ProblemRun[] runs = {
      new ProblemRun("GD", Color.BLUE, gd, ProblemRun.PlotType.Line),
      new ProblemRun("LBFGS", Color.GREEN, lbfgs, ProblemRun.PlotType.Line)
    };
    PlotCanvas iterPlot = log.code(() -> {
      return TestUtil.compare("Model Convergence vs Iteration", runs);
    });
    PlotCanvas timePlot = log.code(() -> {
      return TestUtil.compareTime("Model Convergence vs Time", runs);
    });
    double fitness = Stream.concat(gd.stream(), lbfgs.stream()).mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    return new TestResult(iterPlot, timePlot, fitness);
  }
  
  /**
   * Gets trainable.
   *
   * @param network_gd the network gd
   * @param testInput  the test input
   * @return the trainable
   */
  public ArrayTrainable getTrainable(PipelineNetwork network_gd, Tensor... testInput) {
    return new ArrayTrainable(new Tensor[][]{testInput}, network_gd);
  }
  
  /**
   * Test complete learning test result.
   *
   * @param log            the log
   * @param component      the component
   * @param random         the random
   * @param inputPrototype the input prototype
   * @return the test result
   */
  public TestResult testCompleteLearning(NotebookOutput log, NNLayer component, Random random, Tensor[] inputPrototype) {
    NNLayer network_target = shuffle(random, component.copy()).freeze();
    Tensor[] testInput = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    log.p("In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.code(() -> {
      return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    log.p("We simultaneously regress this target input:");
    log.code(() -> {
      return Arrays.stream(testInput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    log.p("Which produces the following output:");
    Tensor targetOutput = GpuController.call(ctx -> {
      return network_target.eval(ctx, NNResult.singleResultArray(testInput)).getData().get(0);
    });
    log.code(() -> {
      return Stream.of(targetOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    Tensor[] input_gd = shuffle(random, Arrays.stream(inputPrototype).map(t -> t.copy()));
    Tensor[] input_lbgfs = Arrays.stream(input_gd).map(t -> t.copy()).toArray(i -> new Tensor[i]);
    boolean[] mask = new boolean[input_gd.length];
    for (int i = 0; i < mask.length; i++) mask[i] = true;
    PipelineNetwork network_gd = new PipelineNetwork(1);
    network_gd.add(new MeanSqLossLayer(),
      network_gd.add(shuffle(random, component.copy()), network_gd.getInput(0)),
      network_gd.constValue(targetOutput));
    DAGNetwork network_lbfgs = network_gd.copy();
    List<StepRecord> gd = trainCjGD(log, getTrainable(network_gd, input_gd).setMask(mask));
    if (gd.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
      log.p("And regressed input:");
      log.code(() -> {
        return Arrays.stream(input_gd).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
      log.p("Which produces the following output:");
      Tensor regressedOutput = GpuController.call(ctx -> {
        return network_gd.eval(ctx, NNResult.singleResultArray(input_gd)).getData().get(0);
      });
      log.code(() -> {
        return Stream.of(regressedOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    List<StepRecord> lbfgs = trainLBFGS(log, new ArrayTrainable(new Tensor[][]{input_lbgfs}, network_lbfgs).setMask(mask));
    if (lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
      log.p("And regressed input:");
      log.code(() -> {
        return Arrays.stream(input_lbgfs).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
      log.p("Which produces the following output:");
      Tensor regressedOutput = GpuController.call(ctx -> {
        return network_lbfgs.eval(ctx, NNResult.singleResultArray(input_lbgfs)).getData().get(0);
      });
      log.code(() -> {
        return Stream.of(regressedOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    
    ProblemRun[] runs = {
      new ProblemRun("GD", Color.BLUE, gd, ProblemRun.PlotType.Line),
      new ProblemRun("LBFGS", Color.GREEN, lbfgs, ProblemRun.PlotType.Line)
    };
    PlotCanvas iterPlot = log.code(() -> {
      return TestUtil.compare("Integrated Convergence vs Iteration", runs);
    });
    PlotCanvas timePlot = log.code(() -> {
      return TestUtil.compareTime("Integrated Convergence vs Time", runs);
    });
    double fitness = Stream.concat(gd.stream(), lbfgs.stream()).mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    return new TestResult(iterPlot, timePlot, fitness);
  }
  
  /**
   * Test.
   *
   * @param log            the log
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public Double test(NotebookOutput log, final NNLayer component, final Tensor... inputPrototype) {
    boolean testModel = !component.state().isEmpty();
    if (testModel && isZero(component.state().stream().flatMapToDouble(x1 -> Arrays.stream(x1)))) {
      throw new AssertionError("Weights are all zero?");
    }
    if (isZero(Arrays.stream(inputPrototype).flatMapToDouble(x -> Arrays.stream(x.getData())))) {
      throw new AssertionError("Inputs are all zero?");
    }
    Random random = new Random();
    boolean testInput = Arrays.stream(inputPrototype).anyMatch(x -> x.dim() > 0);
    TestResult inputLearning;
    if (testInput) {
      log.h3("Input Learning");
      inputLearning = testInputLearning(log, component, random, inputPrototype);
    }
    else {
      inputLearning = null;
    }
    TestResult modelLearning;
    if (testModel) {
      log.h3("Model Learning");
      modelLearning = testModelLearning(log, component, random, inputPrototype);
    }
    else {
      modelLearning = null;
    }
    TestResult completeLearning;
    if (testInput && testModel) {
      log.h3("Composite Learning");
      completeLearning = testCompleteLearning(log, component, random, inputPrototype);
    }
    else {
      completeLearning = null;
    }
    log.code(() -> {
      return grid(inputLearning, modelLearning, completeLearning);
    });
    return Stream.of(inputLearning, modelLearning, completeLearning).filter(x -> x != null).mapToDouble(x -> x.value).filter(Double::isFinite).min().orElse(Double.NaN);
  }
  
  /**
   * Train cj gd list.
   *
   * @param log       the log
   * @param trainable the trainable
   * @return the list
   */
  public List<StepRecord> trainCjGD(NotebookOutput log, Trainable trainable) {
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
  public List<StepRecord> trainLBFGS(NotebookOutput log, Trainable trainable) {
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
   * Shuffle tensor [ ].
   *
   * @param random the randomize
   * @param copy   the copy
   * @return the tensor [ ]
   */
  private Tensor[] shuffle(Random random, Stream<Tensor> copy) {
    return copy
      .map(tensor -> {
        randomizationMode.shuffle(random, tensor.getData());
        return tensor;
      }).toArray(i -> new Tensor[i]);
  }
  
  /**
   * Shuffle nn layer.
   *
   * @param random        the randomize
   * @param testComponent the test component
   * @return the nn layer
   */
  private NNLayer shuffle(Random random, NNLayer testComponent) {
    testComponent.state().forEach(buffer -> {
      randomizationMode.shuffle(random, buffer);
    });
    return testComponent;
  }
  
  /**
   * Gets randomization mode.
   *
   * @return the randomization mode
   */
  public RandomizationMode getRandomizationMode() {
    return randomizationMode;
  }
  
  /**
   * Sets randomization mode.
   *
   * @param randomizationMode the randomization mode
   * @return the randomization mode
   */
  public ComponentTest setRandomizationMode(RandomizationMode randomizationMode) {
    this.randomizationMode = randomizationMode;
    return this;
  }
  
  /**
   * Grid j panel.
   *
   * @param inputLearning    the input learning
   * @param modelLearning    the model learning
   * @param completeLearning the complete learning
   * @return the j panel
   */
  public JPanel grid(TestResult inputLearning, TestResult modelLearning, TestResult completeLearning) {
    int rows = 0;
    if (inputLearning != null) rows++;
    if (modelLearning != null) rows++;
    if (completeLearning != null) rows++;
    GridLayout layout = new GridLayout(rows, 2, 0, 0);
    JPanel jPanel = new JPanel(layout);
    jPanel.setSize(1200, 400 * rows);
    if (inputLearning != null) {
      jPanel.add(inputLearning.iterPlot == null ? new JPanel() : inputLearning.iterPlot);
      jPanel.add(inputLearning.timePlot == null ? new JPanel() : inputLearning.timePlot);
    }
    if (modelLearning != null) {
      jPanel.add(modelLearning.iterPlot == null ? new JPanel() : modelLearning.iterPlot);
      jPanel.add(modelLearning.timePlot == null ? new JPanel() : modelLearning.timePlot);
    }
    if (completeLearning != null) {
      jPanel.add(completeLearning.iterPlot == null ? new JPanel() : completeLearning.iterPlot);
      jPanel.add(completeLearning.timePlot == null ? new JPanel() : completeLearning.timePlot);
    }
    return jPanel;
  }
  
  /**
   * The type Test result.
   */
  public static class TestResult {
    /**
     * The Time plot.
     */
    PlotCanvas timePlot;
    /**
     * The Iter plot.
     */
    PlotCanvas iterPlot;
    /**
     * The Value.
     */
    double value;
    
    /**
     * Instantiates a new Test result.
     *
     * @param iterPlot the iter plot
     * @param timePlot the time plot
     * @param value    the value
     */
    public TestResult(PlotCanvas iterPlot, PlotCanvas timePlot, double value) {
      this.timePlot = timePlot;
      this.iterPlot = iterPlot;
      this.value = value;
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
  public ComponentTest setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  /**
   * The enum Randomization mode.
   */
  public enum RandomizationMode {
    /**
     * The Permute.
     */
    Permute {
      @Override
      public void shuffle(Random random, double[] buffer) {
        for (int i = 0; i < buffer.length; i++) {
          int j = random.nextInt(buffer.length);
          double v = buffer[i];
          buffer[i] = buffer[j];
          buffer[j] = v;
        }
      }
    }, /**
     * The Permute duplicates.
     */
    PermuteDuplicates {
        @Override
        public void shuffle(Random random, double[] buffer) {
          Permute.shuffle(random, buffer);
          for (int i = 0; i < buffer.length; i++) {
            buffer[i] = buffer[random.nextInt(buffer.length)];
          }
        }
      }, /**
     * The Random.
     */
    Random {
        @Override
        public void shuffle(Random random, double[] buffer) {
          for (int i = 0; i < buffer.length; i++) {
            buffer[i] = 2 * (random.nextDouble() - 0.5);
          }
        }
      };
  
    /**
     * Shuffle.
     *
     * @param random the randomize
     * @param buffer the buffer
     */
    public abstract void shuffle(Random random, double[] buffer);
  }
}
