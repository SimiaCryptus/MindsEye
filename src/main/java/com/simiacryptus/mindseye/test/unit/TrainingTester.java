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
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
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
  public ResultType getResultType(final List<StepRecord> lbfgsmin) {
    return Math.abs(min(lbfgsmin)) < 1e-9 ? ResultType.Converged : ResultType.NonConverged;
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
   * @param zeroTol the zero tol
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
    if (targetOutput.length != random_in.length) return null;
    final Tensor[][] trainingInput = buildInput(random_in, targetOutput);
    final boolean[] mask = buildMask(inputPrototype);
    return eval(log, component, random, inputPrototype, trainingInput, mask);
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
    final Tensor[][] random_in = shuffleCopy(random, inputPrototype);
    if (targetOutput.length != random_in.length) return null;
    final Tensor[][] trainingInput = buildInput(random_in, targetOutput);
    final boolean[] mask = buildMask(inputPrototype);
    return eval(log, shuffleCopy, random, inputPrototype, trainingInput, mask);
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
    if (targetOutput.length != testInput.length) return null;
    final Tensor[][] trainingInput = buildInput(testInput, targetOutput);
    return eval(log, component, random, inputPrototype, trainingInput);
  }
  
  public boolean[] buildMask(Tensor[] inputPrototype) {
    final boolean[] mask = new boolean[inputPrototype.length + 1];
    for (int i = 0; i < inputPrototype.length; i++) {
      mask[i] = true;
    }
    return mask;
  }
  
  public Tensor[][] buildInput(Tensor[][] testInput, Tensor[] targetOutput) {
    return IntStream.range(0, testInput.length).mapToObj(i ->
      Stream.concat(
        Arrays.stream(testInput[i]),
        Stream.of(targetOutput[i])
      ).toArray(j -> new Tensor[j])
    ).toArray(j -> new Tensor[j][]);
  }
  
  public TestResult eval(NotebookOutput log, NNLayer component, Random random, Tensor[] inputPrototype, Tensor[][] trainingInput, boolean... mask) {
    final PipelineNetwork trainingNetwork1 = new PipelineNetwork(inputPrototype.length + 1);
    trainingNetwork1.add(new MeanSqLossLayer(),
      trainingNetwork1.add(shuffle(random, component.copy()), IntStream.range(0, inputPrototype.length).mapToObj(i -> trainingNetwork1.getInput(i)).toArray(i -> new DAGNode[i])),
      trainingNetwork1.getInput(inputPrototype.length));
    PipelineNetwork trainingNetwork = trainingNetwork1;
    final List<StepRecord> gd = train(log, copy(trainingInput), trainingNetwork.copy(), this::trainCjGD, mask);
    final List<StepRecord> lbfgs = train(log, copy(trainingInput), trainingNetwork.copy(), this::trainLBFGS, mask);
    final ProblemRun[] runs = {
      new ProblemRun("GD", Color.BLUE, gd, ProblemRun.PlotType.Line),
      new ProblemRun("LBFGS", Color.GREEN, lbfgs, ProblemRun.PlotType.Line)
    };
    ProblemResult result = new ProblemResult();
    result.put("GD", new TrainingResult(getResultType(gd), min(gd)));
    result.put("LBFGS", new TrainingResult(getResultType(lbfgs), min(lbfgs)));
    final PlotCanvas iterPlot = log.code(() -> {
      return TestUtil.compare("Integrated Convergence vs Iteration", runs);
    });
    final PlotCanvas timePlot = log.code(() -> {
      return TestUtil.compareTime("Integrated Convergence vs Time", runs);
    });
    return new TestResult(iterPlot, timePlot, result);
  }
  
  public double min(List<StepRecord> history) {
    return history.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
  }
  
  public Tensor[][] copy(Tensor[][] input_gd) {
    return Arrays.stream(input_gd)
      .map(t -> Arrays.stream(t).map(v -> v.copy()).toArray(i -> new Tensor[i]))
      .toArray(i -> new Tensor[i][]);
  }
  
  private List<StepRecord> train(NotebookOutput log, Tensor[][] data, NNLayer network, BiFunction<NotebookOutput, Trainable, List<StepRecord>> opt, boolean... mask) {
    ArrayTrainable trainable = new ArrayTrainable(data, network);
    if (0 < mask.length) trainable.setMask(mask);
    List<StepRecord> history = opt.apply(log, trainable);
    if (history.stream().mapToDouble(x -> x.fitness).min().orElse(1) > 1e-5) {
      if (!network.isFrozen()) {
        log.p("This training run resulted in the following configuration:");
        log.code(() -> {
          return network.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
        });
      }
      if (0 < mask.length) {
        log.p("And regressed input:");
        log.code(() -> {
          return Arrays.stream(data)
            .flatMap(x -> Arrays.stream(x))
            .map(x -> x.prettyPrint())
            .reduce((a, b) -> a + "\n" + b)
            .orElse("");
        });
      }
      log.p("To produce the following output:");
      final Tensor regressedOutput = GpuController.call(ctx -> {
        return network.eval(ctx, NNResult.batchResultArray(data)).getData().get(0);
      });
      log.code(() -> {
        return Stream.of(regressedOutput).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
      });
    }
    else {
      log.p("Training Converged");
    }
    return history;
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
    try {
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
    } catch (Throwable e) {
    }
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
    try {
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
    } catch (Throwable e) {
    }
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
    Map<String, TrainingResult> map;
  
    /**
     * Instantiates a new Problem result.
     *
     */
    public ProblemResult() {
      this.map = new HashMap<>();
    }
  
    public ProblemResult put(String key, TrainingResult result) {
      map.put(key, result);
      return this;
    }
  
    @Override
    public String toString() {
      return map.toString();
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
