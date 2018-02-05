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

package com.simiacryptus.mindseye.test.unit;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.BasicTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.opt.orient.RecursiveSubspace;
import com.simiacryptus.mindseye.test.ProblemRun;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;

import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Derivative tester.
 */
public class TrainingTester extends ComponentTestBase<TrainingTester.ComponentResult> {
  /**
   * The Logger.
   */
  static final Logger log = LoggerFactory.getLogger(TrainingTester.class);
  
  private int batches = 3;
  private RandomizationMode randomizationMode = RandomizationMode.Permute;
  private boolean verbose = true;
  private boolean throwExceptions = false;
  
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
        SysOutInterceptor.ORIGINAL_OUT.println(msg);
        log.info(msg);
      }
  
      @Override
      public void onStepComplete(final Step currentPoint) {
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
      }
    };
  }
  
  /**
   * Build input tensor [ ] [ ].
   *
   * @param left  the run input
   * @param right the target output
   * @return the tensor [ ] [ ]
   */
  public static Tensor[][] append(Tensor[][] left, Tensor[] right) {
    return IntStream.range(0, left.length).mapToObj(i ->
                                                      Stream.concat(
                                                        Arrays.stream(left[i]),
                                                        Stream.of(right[i])
                                                                   ).toArray(j -> new Tensor[j])
                                                   ).toArray(j -> new Tensor[j][]);
  }
  
  /**
   * Copy tensor [ ] [ ].
   *
   * @param input_gd the input gd
   * @return the tensor [ ] [ ]
   */
  public static Tensor[][] copy(Tensor[][] input_gd) {
    return Arrays.stream(input_gd)
                 .map(t -> Arrays.stream(t).map(v -> v.copy()).toArray(i -> new Tensor[i]))
                 .toArray(i -> new Tensor[i][]);
  }
  
  /**
   * Pop tensor [ ] [ ].
   *
   * @param data the data
   * @return the tensor [ ] [ ]
   */
  public static Tensor[][] pop(Tensor[][] data) {
    return Arrays.stream(data)
                 .map(t -> Arrays.stream(t).limit(t.length - 1).toArray(i -> new Tensor[i]))
                 .toArray(i -> new Tensor[i][]);
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
   * @param testComponent the run component
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
    printHeader(log);
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
      log.h2("Input Learning");
      inputLearning = testInputLearning(log, component, random, inputPrototype);
    }
    else {
      inputLearning = null;
    }
    TestResult modelLearning;
    if (testModel) {
      log.h2("Model Learning");
      modelLearning = testModelLearning(log, component, random, inputPrototype);
    }
    else {
      modelLearning = null;
    }
    TestResult completeLearning;
    if (testInput && testModel) {
      log.h2("Composite Learning");
      completeLearning = testCompleteLearning(log, component, random, inputPrototype);
    }
    else {
      completeLearning = null;
    }
    log.h2("Results");
    log.code(() -> {
      return grid(inputLearning, modelLearning, completeLearning);
    });
    ComponentResult result = log.code(() -> {
      return new ComponentResult(
        null == inputLearning ? null : inputLearning.value,
        null == modelLearning ? null : modelLearning.value,
        null == completeLearning ? null : completeLearning.value);
    });
    log.setFrontMatterProperty("training_analysis", result.toString());
    if (throwExceptions) {
      assert result.complete.map.values().stream().allMatch(x -> x.type == ResultType.Converged);
      assert result.input.map.values().stream().allMatch(x -> x.type == ResultType.Converged);
      assert result.model.map.values().stream().allMatch(x -> x.type == ResultType.Converged);
    }
    return result;
  }
  
  /**
   * Print header.
   *
   * @param log the log
   */
  protected void printHeader(NotebookOutput log) {
    log.h1("Training Characteristics");
  }
  
  /**
   * Test complete learning run result.
   *
   * @param log            the log
   * @param component      the component
   * @param random         the random
   * @param inputPrototype the input prototype
   * @return the run result
   */
  public TestResult testCompleteLearning(final NotebookOutput log, final NNLayer component, final Random random, final Tensor[] inputPrototype) {
    final NNLayer network_target = shuffle(random, component.copy()).freeze();
    final Tensor[][] input_target = shuffleCopy(random, inputPrototype);
    log.p("In this run, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.code(() -> {
      return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    log.p("We simultaneously regress this target input:");
    log.code(() -> {
      return Arrays.stream(input_target)
                   .flatMap(x -> Arrays.stream(x))
                   .map(x -> x.prettyPrint())
                   .reduce((a, b) -> a + "\n" + b)
                   .orElse("");
    });
    log.p("Which produces the following output:");
    TensorList result = network_target.eval(NNConstant.batchResultArray(input_target)).getData();
    result.stream().forEach(x -> x.addRef());
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    log.code(() -> {
      return Stream.of(output_target).map(x -> x.prettyPrint()).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    //if (output_target.length != inputPrototype.length) return null;
    return trainAll("Integrated Convergence", log,
                    append(shuffleCopy(random, inputPrototype), output_target),
                    shuffle(random, component.copy()),
                    buildMask(inputPrototype.length));
  }
  
  /**
   * Test input learning.
   *
   * @param log            the log
   * @param component      the component
   * @param random         the randomize
   * @param inputPrototype the input prototype
   * @return the run result
   */
  public TestResult testInputLearning(final NotebookOutput log, final NNLayer component, final Random random, final Tensor[] inputPrototype) {
    final NNLayer network = shuffle(random, component.copy()).freeze();
    final Tensor[][] input_target = shuffleCopy(random, inputPrototype);
    log.p("In this run, we use a network to learn this target input, given it's pre-evaluated output:");
    log.code(() -> {
      return Arrays.stream(input_target)
                   .flatMap(x -> Arrays.stream(x))
                   .map(x -> x.prettyPrint())
                   .reduce((a, b) -> a + "\n" + b)
                   .orElse("");
    });
    NNResult[] array = NNConstant.batchResultArray(input_target);
    NNResult eval = network.eval(array);
    TensorList result = eval.getData();
    eval.freeRef();
    for (NNResult nnResult : array) {
      nnResult.getData().freeRef();
      nnResult.freeRef();
    }
    for (Tensor[] tensors : input_target) {
      for (Tensor tensor : tensors) {
        tensor.freeRef();
      }
    }
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    Arrays.stream(output_target).forEach(x -> x.addRef());
    result.freeRef();
    //if (output_target.length != inputPrototype.length) return null;
    Tensor[][] trainingInput = append(shuffleCopy(random, inputPrototype), output_target);
    TestResult testResult = trainAll("Input Convergence", log,
                                     trainingInput,
                                     network,
                                     buildMask(inputPrototype.length));
    network.freeRef();
    Arrays.stream(trainingInput).flatMap(x -> Arrays.stream(x)).forEach(x -> x.freeRef());
    return testResult;
  }
  
  /**
   * Test model learning.
   *
   * @param log            the log
   * @param component      the component
   * @param random         the randomize
   * @param inputPrototype the input prototype
   * @return the run result
   */
  public TestResult testModelLearning(final NotebookOutput log, final NNLayer component, final Random random, final Tensor[] inputPrototype) {
    final NNLayer network_target = shuffle(random, component.copy()).freeze();
    final Tensor[][] input_target = shuffleCopy(random, inputPrototype);
    log.p("In this run, attempt to train a network to emulate a randomized network given an example input/output. The target state is:");
    log.code(() -> {
      return network_target.state().stream().map(Arrays::toString).reduce((a, b) -> a + "\n" + b).orElse("");
    });
    TensorList result = network_target.eval(NNConstant.batchResultArray(input_target)).getData();
    result.stream().forEach(x -> x.addRef());
    final Tensor[] output_target = result.stream().toArray(i -> new Tensor[i]);
    //if (output_target.length != input_target.length) return null;
    return trainAll("Model Convergence", log,
                    append(input_target, output_target),
                    shuffle(random, component.copy()));
  }
  
  /**
   * Min double.
   *
   * @param history the history
   * @return the double
   */
  public double min(List<StepRecord> history) {
    return history.stream().mapToDouble(x -> x.fitness).min().orElse(Double.NaN);
  }
  
  /**
   * Build mask boolean [ ].
   *
   * @param length the length
   * @return the boolean [ ]
   */
  public boolean[] buildMask(int length) {
    final boolean[] mask = new boolean[length + 1];
    for (int i = 0; i < length; i++) {
      mask[i] = true;
    }
    return mask;
  }
  
  /**
   * Train all run result.
   *
   * @param title         the title
   * @param log           the log
   * @param trainingInput the training input
   * @param layer         the layer
   * @param mask          the mask
   * @return the run result
   */
  public TestResult trainAll(String title, NotebookOutput log, Tensor[][] trainingInput, NNLayer layer, boolean... mask) {
    log.h3("Gradient Descent");
    final List<StepRecord> gd = train(log, this::trainGD, layer.copy(), copy(trainingInput), mask);
    log.h3("Conjugate Gradient Descent");
    final List<StepRecord> cjgd = train(log, this::trainCjGD, layer.copy(), copy(trainingInput), mask);
    log.h3("Limited-Memory BFGS");
    final List<StepRecord> lbfgs = train(log, this::trainLBFGS, layer.copy(), copy(trainingInput), mask);
    log.h3("Experimental Optimizer");
    final List<StepRecord> magic = train(log, this::trainMagic, layer.copy(), copy(trainingInput), mask);
    final ProblemRun[] runs = {
      new ProblemRun("GD", gd, Color.GRAY, ProblemRun.PlotType.Line),
      new ProblemRun("CjGD", cjgd, Color.CYAN, ProblemRun.PlotType.Line),
      new ProblemRun("LBFGS", lbfgs, Color.GREEN, ProblemRun.PlotType.Line),
      new ProblemRun("Experimental", magic, Color.MAGENTA, ProblemRun.PlotType.Line)
    };
    ProblemResult result = new ProblemResult();
    result.put("GD", new TrainingResult(getResultType(gd), min(gd)));
    result.put("CjGD", new TrainingResult(getResultType(cjgd), min(cjgd)));
    result.put("LBFGS", new TrainingResult(getResultType(lbfgs), min(lbfgs)));
    result.put("Experimental", new TrainingResult(getResultType(magic), min(magic)));
    if (verbose) {
      final PlotCanvas iterPlot = log.code(() -> {
        return TestUtil.compare(title + " vs Iteration", runs);
      });
      final PlotCanvas timePlot = log.code(() -> {
        return TestUtil.compareTime(title + " vs Time", runs);
      });
      return new TestResult(iterPlot, timePlot, result);
    }
    else {
      final PlotCanvas iterPlot = TestUtil.compare(title + " vs Iteration", runs);
      final PlotCanvas timePlot = TestUtil.compareTime(title + " vs Time", runs);
      return new TestResult(iterPlot, timePlot, result);
    }
  }
  
  private List<StepRecord> train(NotebookOutput log, BiFunction<NotebookOutput, Trainable, List<StepRecord>> opt, NNLayer layer, Tensor[][] data, boolean... mask) {
    try {
      int inputs = data[0].length;
      final PipelineNetwork network = new PipelineNetwork(inputs);
      network.add(new MeanSqLossLayer(),
                  network.add(layer, IntStream.range(0, inputs - 1).mapToObj(i -> network.getInput(i)).toArray(i -> new DAGNode[i])),
                  network.getInput(inputs - 1));
      ArrayTrainable trainable = new ArrayTrainable(data, network);
      if (0 < mask.length) trainable.setMask(mask);
      List<StepRecord> history = opt.apply(log, trainable);
      trainable.freeRef();
      network.freeRef();
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
                         .limit(1)
                         .map(x -> x.prettyPrint())
                         .reduce((a, b) -> a + "\n" + b)
                         .orElse("");
          });
        }
        log.p("To produce the following output:");
        log.code(() -> {
          NNResult[] array = NNConstant.batchResultArray(pop(data));
          NNResult eval = layer.eval(array);
          for (NNResult nnResult : array) {
            nnResult.freeRef();
            nnResult.getData().freeRef();
          }
          TensorList tensorList = eval.getData();
          eval.freeRef();
          String str = tensorList.stream()
                                 .collect(Collectors.toList())
                                 .stream()
                                 .limit(1)
                                 .map(x -> x.prettyPrint())
                                 .reduce((a, b) -> a + "\n" + b)
                                 .orElse("");
          tensorList.freeRef();
          return str;
        });
      }
      else {
        log.p("Training Converged");
      }
      return history;
    } finally {
      for (Tensor[] tensors : data) {
        for (Tensor tensor : tensors) {
          tensor.freeRef();
        }
      }
    }
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
          .runAndFree();
      });
    } catch (Throwable e) {
      if (isThrowExceptions()) throw new RuntimeException(e);
    }
    return history;
  }
  
  /**
   * Train gd list.
   *
   * @param log       the log
   * @param trainable the trainable
   * @return the list
   */
  public List<StepRecord> trainGD(final NotebookOutput log, final Trainable trainable) {
    log.p("First, we train using basic gradient descent method with weak line search conditions.");
    final List<StepRecord> history = new ArrayList<>();
    final TrainingMonitor monitor = TrainingTester.getMonitor(history);
    try {
      log.code(() -> {
        return new IterativeTrainer(trainable)
          .setLineSearchFactory(label -> new ArmijoWolfeSearch())
          .setOrientation(new GradientDescent())
          .setMonitor(monitor)
          .setTimeout(30, TimeUnit.SECONDS)
          .setMaxIterations(250)
          .setTerminateThreshold(0)
          .runAndFree();
      });
    } catch (Throwable e) {
      if (isThrowExceptions()) throw new RuntimeException(e);
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
          .runAndFree();
      });
    } catch (Throwable e) {
      if (isThrowExceptions()) throw new RuntimeException(e);
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
  public List<StepRecord> trainMagic(final NotebookOutput log, final Trainable trainable) {
    log.p("Now we train using an experimental optimizer:");
    final List<StepRecord> history = new ArrayList<>();
    final TrainingMonitor monitor = TrainingTester.getMonitor(history);
    try {
      log.code(() -> {
        return new IterativeTrainer(trainable)
          .setLineSearchFactory(label -> new StaticLearningRate(1.0))
          .setOrientation(new RecursiveSubspace() {
            @Override
            public void train(TrainingMonitor monitor, NNLayer macroLayer) {
              Tensor[][] nullData = {{new Tensor()}};
              BasicTrainable inner = new BasicTrainable(macroLayer);
              ArrayTrainable trainable1 = new ArrayTrainable(inner, nullData);
              inner.freeRef();
              new IterativeTrainer(trainable1)
                .setOrientation(new QQN())
                .setLineSearchFactory(n -> new QuadraticSearch().setCurrentRate(n.equals(QQN.CURSOR_NAME) ? 1.0 : 1e-4))
                .setMonitor(new TrainingMonitor() {
                  @Override
                  public void log(String msg) {
                    monitor.log("\t" + msg);
                  }
                }).setMaxIterations(getIterations()).setIterationsPerSample(getIterations()).runAndFree();
              trainable1.freeRef();
              for (Tensor[] tensors : nullData) {
                for (Tensor tensor : tensors) {
                  tensor.freeRef();
                }
              }
            }
          })
          .setMonitor(monitor)
          .setTimeout(30, TimeUnit.SECONDS)
          .setIterationsPerSample(100)
          .setMaxIterations(250)
          .setTerminateThreshold(0)
          .runAndFree();
      });
    } catch (Throwable e) {
      if (isThrowExceptions()) throw new RuntimeException(e);
    }
    return history;
  }
  
  /**
   * Is throw exceptions boolean.
   *
   * @return the boolean
   */
  public boolean isThrowExceptions() {
    return throwExceptions;
  }
  
  /**
   * Sets throw exceptions.
   *
   * @param throwExceptions the throw exceptions
   * @return the throw exceptions
   */
  public TrainingTester setThrowExceptions(boolean throwExceptions) {
    this.throwExceptions = throwExceptions;
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
      return String.format("{\"input\":%s, \"model\":%s, \"complete\":%s}", input, model, complete);
    }
  }
  
  /**
   * The type Problem result.
   */
  public static class ProblemResult {
    /**
     * The Map.
     */
    Map<String, TrainingResult> map;
  
    /**
     * Instantiates a new Problem result.
     */
    public ProblemResult() {
      this.map = new HashMap<>();
    }
  
    /**
     * Put problem result.
     *
     * @param key    the key
     * @param result the result
     * @return the problem result
     */
    public ProblemResult put(String key, TrainingResult result) {
      map.put(key, result);
      return this;
    }
  
    @Override
    public String toString() {
      return map.entrySet().stream().map(e -> {
        return String.format("\"%s\": %s", e.getKey(), e.getValue().toString());
      }).reduce((a, b) -> a + ", " + b).get();
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
      return String.format("{\"type\":\"%s\", value:%s}", type, value);
    }
  }
  
  @Override
  public String toString() {
    return "TrainingTester{" +
      "batches=" + batches +
      ", randomizationMode=" + randomizationMode +
      ", verbose=" + verbose +
      ", throwExceptions=" + throwExceptions +
      '}';
  }
}
