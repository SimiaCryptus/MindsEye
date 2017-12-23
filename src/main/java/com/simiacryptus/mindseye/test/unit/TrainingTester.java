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
import java.io.PrintStream;
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
public class TrainingTester implements ComponentTest<TrainingTester.ComponentResult> {
  
  private static final PrintStream originalOut = System.out;
  private int batches = 3;
  private RandomizationMode randomizationMode = RandomizationMode.Permute;
  private boolean verbose = true;
  
  /**
   * Instantiates a new Learning tester.
   */
  public TrainingTester() {
  }
  
  /**
   * Gets monitor.
   *
   * @param history the history
   * @return the monitor
   */
  public static TrainingMonitor getMonitor(final List<StepRecord> history) {
    return new TrainingMonitor() {
      @Override
      public void log(final String msg) {
        originalOut.println(msg);
        System.out.println(msg);
      }
  
      @Override
      public void onStepComplete(final Step currentPoint) {
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
      }
    };
  }
  
  /**
   * Gets batches.
   *
   * @return the batches
   */
  public int getBatches() {
    return batches;
  }
  
  /**
   * Sets batches.
   *
   * @param batches the batches
   */
  public void setBatches(final int batches) {
    this.batches = batches;
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
  public TrainingTester setRandomizationMode(final RandomizationMode randomizationMode) {
    this.randomizationMode = randomizationMode;
    return this;
  }
  
  /**
   * Gets result type.
   *
   * @param lbfgsmin the lbfgsmin
   * @return the result type
   */
  public ResultType getResultType(final double lbfgsmin) {
    return Math.abs(lbfgsmin) < 1e-9 ? ResultType.Converged : ResultType.NonConverged;
  }
  
  /**
   * Grid j panel.
   *
   * @param inputLearning    the input learning
   * @param modelLearning    the model learning
   * @param completeLearning the complete learning
   * @return the j panel
   */
  public JPanel grid(final TestResult inputLearning, final TestResult modelLearning, final TestResult completeLearning) {
    int rows = 0;
    if (inputLearning != null) {
      rows++;
    }
    if (modelLearning != null) {
      rows++;
    }
    if (completeLearning != null) {
      rows++;
    }
    final GridLayout layout = new GridLayout(rows, 2, 0, 0);
    final JPanel jPanel = new JPanel(layout);
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
  public TrainingTester setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  /**
   * Is zero boolean.
   *
   * @param stream the stream
   * @return the boolean
   */
  public boolean isZero(final DoubleStream stream) {
    return isZero(stream, 1e-14);
  }
  
  /**
   * Is zero boolean.
   *
   * @param stream  the stream
   * @param zeroTol
   * @return the boolean
   */
  public boolean isZero(final DoubleStream stream, double zeroTol) {
    final double[] array = stream.toArray();
    if (array.length == 0) return false;
    return Arrays.stream(array).map(x -> Math.abs(x)).sum() < zeroTol;
  }
  
  /**
   * Shuffle nn layer.
   *
   * @param random        the randomize
   * @param testComponent the test component
   * @return the nn layer
   */
  private NNLayer shuffle(final Random random, final NNLayer testComponent) {
    testComponent.state().forEach(buffer -> {
      randomizationMode.shuffle(random, buffer);
    });
    return testComponent;
  }
  
  /**
   * Shuffle tensor [ ].
   *
   * @param random the randomize
   * @param copy   the copy
   * @return the tensor [ ]
   */
  private Tensor[][] shuffleCopy(final Random random, final Tensor... copy) {
    return IntStream.range(0, getBatches()).mapToObj(i -> {
      return Arrays.stream(copy).map(tensor -> {
        final Tensor cpy = tensor.copy();
        randomizationMode.shuffle(random, cpy.getData());
        return cpy;
      }).toArray(j -> new Tensor[j]);
    }).toArray(i -> new Tensor[i][]);
  }
  
  /**
   * Test.
   *
   * @param log            the log
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  @Override
  public ComponentResult test(final NotebookOutput log, final NNLayer component, final Tensor... inputPrototype) {
    final boolean testModel = !component.state().isEmpty();
    if (testModel && isZero(component.state().stream().flatMapToDouble(x1 -> Arrays.stream(x1)))) {
      throw new AssertionError("Weights are all zero?");
    }
    if (isZero(Arrays.stream(inputPrototype).flatMapToDouble(x -> Arrays.stream(x.getData())))) {
      throw new AssertionError("Inputs are all zero?");
    }
    final Random random = new Random();
    final boolean testInput = Arrays.stream(inputPrototype).anyMatch(x -> x.dim() > 0);
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
    return log.code(() -> {
      return new ComponentResult(
        null == inputLearning ? null : inputLearning.value,
        null == modelLearning ? null : modelLearning.value,
        null == completeLearning ? null : completeLearning.value);
    });
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
  public TestResult testCompleteLearning(final NotebookOutput log, final NNLayer component, final Random random, final Tensor[] inputPrototype) {
    final NNLayer network_target = shuffle(random, component.copy()).freeze();
    final Tensor[][] testInput = shuffleCopy(random, inputPrototype);
    log.p("In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.code(() -> {
      return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    log.p("We simultaneously regress this target input:");
    log.code(() -> {
      return Arrays.stream(testInput)
        .flatMap(x -> Arrays.stream(x))
        .map(x -> x.prettyPrint())
        .reduce((a, b) -> a + "\n" + b)
        .orElse("");
    });
    log.p("Which produces the following output:");
    final Tensor[] targetOutput = GpuController.call(ctx -> {
      return network_target.eval(ctx, NNResult.batchResultArray(testInput)).getData();
    }).stream().toArray(i -> new Tensor[i]);
    log.code(() -> {
      return Stream.of(targetOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    final Tensor[][] random_in = shuffleCopy(random, inputPrototype);
    final Tensor[][] input_gd = IntStream.range(0, random_in.length).mapToObj(i ->
      Stream.concat(
        Arrays.stream(random_in[i]),
        Stream.of(targetOutput[i])
      ).toArray(j -> new Tensor[j])
    ).toArray(j -> new Tensor[j][]);
    final Tensor[][] input_lbgfs = Arrays.stream(input_gd)
      .map(t -> Arrays.stream(t).map(v -> v.copy()).toArray(i -> new Tensor[i]))
      .toArray(i -> new Tensor[i][]);
    final boolean[] mask = new boolean[input_gd.length];
    for (int i = 0; i < mask.length; i++) {
      mask[i] = true;
    }
    final PipelineNetwork network_gd = new PipelineNetwork(inputPrototype.length + 1);
    network_gd.add(new MeanSqLossLayer(),
      network_gd.add(shuffle(random, component.copy()), network_gd.getInput(0)),
      network_gd.getInput(inputPrototype.length));
    final DAGNetwork network_lbfgs = network_gd.copy();
    final List<StepRecord> gd = trainCjGD(log, new ArrayTrainable(input_gd, network_gd).setMask(mask));
    if (gd.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
      log.p("And regressed input:");
      log.code(() -> {
        return Arrays.stream(input_gd)
          .flatMap(x -> Arrays.stream(x))
          .map(x -> x.prettyPrint())
          .reduce((a, b) -> a + "\n" + b)
          .orElse("");
      });
      log.p("Which produces the following output:");
      final Tensor regressedOutput = GpuController.call(ctx -> {
        return network_gd.eval(ctx, NNResult.batchResultArray(input_gd)).getData().get(0);
      });
      log.code(() -> {
        return Stream.of(regressedOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    final List<StepRecord> lbfgs = trainLBFGS(log, new ArrayTrainable(input_lbgfs, network_lbfgs).setMask(mask));
    if (lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
      log.p("And regressed input:");
      log.code(() -> {
        return Arrays.stream(input_lbgfs)
          .flatMap(x -> Arrays.stream(x))
          .map(x -> x.prettyPrint())
          .reduce((a, b) -> a + "\n" + b)
          .orElse("");
      });
      log.p("Which produces the following output:");
      final Tensor regressedOutput = GpuController.call(ctx -> {
        return network_lbfgs.eval(ctx, NNResult.batchResultArray(input_lbgfs)).getData().get(0);
      });
      log.code(() -> {
        return Stream.of(regressedOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    
    final ProblemRun[] runs = {
      new ProblemRun("GD", Color.BLUE, gd, ProblemRun.PlotType.Line),
      new ProblemRun("LBFGS", Color.GREEN, lbfgs, ProblemRun.PlotType.Line)
    };
    final PlotCanvas iterPlot = log.code(() -> {
      return TestUtil.compare("Integrated Convergence vs Iteration", runs);
    });
    final PlotCanvas timePlot = log.code(() -> {
      return TestUtil.compareTime("Integrated Convergence vs Time", runs);
    });
    final double gdmin = gd.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    final double lbfgsmin = lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    return new TestResult(iterPlot, timePlot, new ProblemResult(
      new TrainingResult(getResultType(gdmin), gdmin),
      new TrainingResult(getResultType(lbfgsmin), lbfgsmin)
    ));
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
  public TestResult testInputLearning(final NotebookOutput log, final NNLayer component, final Random random, final Tensor[] inputPrototype) {
    final NNLayer shuffleCopy = shuffle(random, component.copy()).freeze();
    final Tensor[][] input_target = shuffleCopy(random, inputPrototype);
    log.p("In this test, we use a network to learn this target input, given it's pre-evaluated output:");
    log.code(() -> {
      return Arrays.stream(input_target)
        .flatMap(x -> Arrays.stream(x))
        .map(x -> x.prettyPrint())
        .reduce((a, b) -> a + "\n" + b)
        .orElse("");
    });
    final Tensor[] targetOutput = GpuController.call(ctx -> {
      return shuffleCopy.eval(ctx, NNResult.batchResultArray(input_target)).getData();
    }).stream().toArray(i -> new Tensor[i]);
    final PipelineNetwork network = new PipelineNetwork(inputPrototype.length + 1);
    network.add(new MeanSqLossLayer(),
      network.add(shuffleCopy, IntStream.range(0, inputPrototype.length).mapToObj(i -> network.getInput(0)).toArray(i -> new DAGNode[i])),
      network.getInput(inputPrototype.length));
    final Tensor[][] random_in = shuffleCopy(random, inputPrototype);
    if (targetOutput.length != random_in.length) throw new AssertionError();
    final Tensor[][] input_gd = IntStream.range(0, random_in.length).mapToObj(i ->
      Stream.concat(
        Arrays.stream(random_in[i]),
        Stream.of(targetOutput[i])
      ).toArray(j -> new Tensor[j])
    ).toArray(j -> new Tensor[j][]);
    final Tensor[][] input_lbgfs = Arrays.stream(input_gd)
      .map(t -> Arrays.stream(t).map(x -> x.copy()).toArray(i -> new Tensor[i]))
      .toArray(i -> new Tensor[i][]);
    final boolean[] mask = new boolean[input_gd.length];
    for (int i = 0; i < mask.length; i++) {
      mask[i] = true;
    }
    final List<StepRecord> gd = trainCjGD(log, new ArrayTrainable(input_gd, network).setMask(mask));
    if (gd.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following regressed input:");
      log.code(() -> {
        return Arrays.stream(input_gd)
          .flatMap(x -> Arrays.stream(x))
          .map(x -> x.prettyPrint())
          .reduce((a, b) -> a + "\n" + b)
          .orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    final List<StepRecord> lbfgs = trainLBFGS(log, new ArrayTrainable(input_lbgfs, network).setMask(true));
    if (lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following regressed input:");
      log.code(() -> {
        return Arrays.stream(input_lbgfs)
          .flatMap(x -> Arrays.stream(x))
          .map(x -> x.prettyPrint())
          .reduce((a, b) -> a + "\n" + b)
          .orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    final ProblemRun[] runs = {
      new ProblemRun("GD", Color.BLUE, gd, ProblemRun.PlotType.Line),
      new ProblemRun("LBFGS", Color.GREEN, lbfgs, ProblemRun.PlotType.Line)
    };
    final PlotCanvas iterPlot = log.code(() -> {
      return TestUtil.compare("Input Convergence vs Iteration", runs);
    });
    final PlotCanvas timePlot = log.code(() -> {
      return TestUtil.compareTime("Input Convergence vs Time", runs);
    });
    final double gdmin = gd.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    final double lbfgsmin = lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    return new TestResult(iterPlot, timePlot, new ProblemResult(
      new TrainingResult(getResultType(gdmin), gdmin),
      new TrainingResult(getResultType(lbfgsmin), lbfgsmin)
    ));
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
  public TestResult testModelLearning(final NotebookOutput log, final NNLayer component, final Random random, final Tensor[] inputPrototype) {
    final NNLayer network_target = shuffle(random, component.copy()).freeze();
    final Tensor[][] testInput = shuffleCopy(random, inputPrototype);
    log.p("In this test, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.code(() -> {
      return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    final Tensor[] targetOutput = GpuController.call(ctx -> {
      return network_target.eval(ctx, NNResult.batchResultArray(testInput)).getData();
    }).stream().toArray(i -> new Tensor[i]);
    if (targetOutput.length != testInput.length) throw new AssertionError();
    final Tensor[][] appended = IntStream.range(0, testInput.length).mapToObj(i ->
      Stream.concat(
        Arrays.stream(testInput[i]),
        Stream.of(targetOutput[i])
      ).toArray(j -> new Tensor[j])
    ).toArray(j -> new Tensor[j][]);
    final PipelineNetwork network_gd = new PipelineNetwork(inputPrototype.length + 1);
    network_gd.add(new MeanSqLossLayer(),
      network_gd.add(shuffle(random, component.copy()), network_gd.getInput(0)),
      network_gd.getInput(inputPrototype.length));
    final List<StepRecord> gd = trainCjGD(log, new ArrayTrainable(appended, network_gd));
    if (gd.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network_gd.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    final PipelineNetwork network_lbfgs = new PipelineNetwork(inputPrototype.length + 1);
    network_lbfgs.add(new MeanSqLossLayer(),
      network_lbfgs.add(shuffle(random, component.copy()), network_lbfgs.getInput(0)),
      network_lbfgs.getInput(inputPrototype.length));
    final List<StepRecord> lbfgs = trainLBFGS(log, new ArrayTrainable(appended, network_lbfgs));
    if (lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      log.p("This training run resulted in the following configuration:");
      log.code(() -> {
        return network_lbfgs.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
  
    final ProblemRun[] runs = {
      new ProblemRun("GD", Color.BLUE, gd, ProblemRun.PlotType.Line),
      new ProblemRun("LBFGS", Color.GREEN, lbfgs, ProblemRun.PlotType.Line)
    };
    final PlotCanvas iterPlot = log.code(() -> {
      return TestUtil.compare("Model Convergence vs Iteration", runs);
    });
    final PlotCanvas timePlot = log.code(() -> {
      return TestUtil.compareTime("Model Convergence vs Time", runs);
    });
    final double gdmin = gd.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    final double lbfgsmin = lbfgs.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
    return new TestResult(iterPlot, timePlot, new ProblemResult(
      new TrainingResult(getResultType(gdmin), gdmin),
      new TrainingResult(getResultType(lbfgsmin), lbfgsmin)
    ));
  }
  
  /**
   * Train cj gd list.
   *
   * @param log       the log
   * @param trainable the trainable
   * @return the list
   */
  public List<StepRecord> trainCjGD(final NotebookOutput log, final Trainable trainable) {
    log.p("First, we use a conjugate gradient descent method, which converges the fastest for purely linear functions.");
    final List<StepRecord> history = new ArrayList<>();
    final TrainingMonitor monitor = TrainingTester.getMonitor(history);
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
  public List<StepRecord> trainLBFGS(final NotebookOutput log, final Trainable trainable) {
    log.p("Next, we run the same optimization using L-BFGS, which is nearly ideal for purely second-order or quadratic functions.");
    final List<StepRecord> history = new ArrayList<>();
    final TrainingMonitor monitor = TrainingTester.getMonitor(history);
    log.code(() -> {
      return new IterativeTrainer(trainable)
        .setLineSearchFactory(label -> new ArmijoWolfeSearch())
        .setOrientation(new LBFGS())
        .setMonitor(monitor)
        .setTimeout(30, TimeUnit.SECONDS)
        .setIterationsPerSample(100)
        .setMaxIterations(250)
        .setTerminateThreshold(0)
        .run();
    });
    return history;
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
      public void shuffle(final Random random, final double[] buffer) {
        for (int i = 0; i < buffer.length; i++) {
          final int j = random.nextInt(buffer.length);
          final double v = buffer[i];
          buffer[i] = buffer[j];
          buffer[j] = v;
        }
      }
    }, /**
     * The Permute duplicates.
     */
    PermuteDuplicates {
        @Override
        public void shuffle(final Random random, final double[] buffer) {
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
        public void shuffle(final Random random, final double[] buffer) {
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
  
  /**
   * The enum Result type.
   */
  public enum ResultType {
    /**
     * Converged result type.
     */
    Converged,
    /**
     * Non converged result type.
     */
    NonConverged
  }
  
  /**
   * The type Component result.
   */
  public static class ComponentResult {
    /**
     * The Complete.
     */
    ProblemResult complete;
    /**
     * The Input.
     */
    ProblemResult input;
    /**
     * The Model.
     */
    ProblemResult model;
  
    /**
     * Instantiates a new Component result.
     *
     * @param input    the input
     * @param model    the model
     * @param complete the complete
     */
    public ComponentResult(final ProblemResult input, final ProblemResult model, final ProblemResult complete) {
      this.input = input;
      this.model = model;
      this.complete = complete;
    }

    @Override
    public String toString() {
      return String.format("ComponentResult{input=%s, model=%s, complete=%s}", input, model, complete);
    }
  }
  
  /**
   * The type Problem result.
   */
  public static class ProblemResult {
    /**
     * The Cjgd.
     */
    TrainingResult cjgd;
    /**
     * The Lbfgs.
     */
    TrainingResult lbfgs;
    
    /**
     * Instantiates a new Problem result.
     *
     * @param cjgd  the cjgd
     * @param lbfgs the lbfgs
     */
    public ProblemResult(final TrainingResult cjgd, final TrainingResult lbfgs) {
      this.cjgd = cjgd;
      this.lbfgs = lbfgs;
    }
    
    @Override
    public String toString() {
      return String.format("ProblemResult{cjgd=%s, lbfgs=%s}", cjgd, lbfgs);
    }
  }
  
  /**
   * The type Test result.
   */
  public static class TestResult {
    /**
     * The Iter plot.
     */
    PlotCanvas iterPlot;
    /**
     * The Time plot.
     */
    PlotCanvas timePlot;
    /**
     * The Value.
     */
    ProblemResult value;
  
    /**
     * Instantiates a new Test result.
     *
     * @param iterPlot the iter plot
     * @param timePlot the time plot
     * @param value    the value
     */
    public TestResult(final PlotCanvas iterPlot, final PlotCanvas timePlot, final ProblemResult value) {
      this.timePlot = timePlot;
      this.iterPlot = iterPlot;
      this.value = value;
    }
  }
  
  /**
   * The type Training result.
   */
  public static class TrainingResult {
    /**
     * The Type.
     */
    ResultType type;
    /**
     * The Value.
     */
    double value;
    
    /**
     * Instantiates a new Training result.
     *
     * @param type  the type
     * @param value the value
     */
    public TrainingResult(final ResultType type, final double value) {
      this.type = type;
      this.value = value;
    }
    
    @Override
    public String toString() {
      return String.format("TrainingResult{type=%s, value=%s}", type, value);
    }
  }
}
