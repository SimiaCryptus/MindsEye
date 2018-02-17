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

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.java.PlaceholderLayer;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Derivative tester.
 */
public class BatchDerivativeTester extends ComponentTestBase<ToleranceStatistics> {
  /**
   * The Logger.
   */
  static final Logger log = LoggerFactory.getLogger(BatchDerivativeTester.class);
  
  /**
   * The Probe size.
   */
  public final double probeSize;
  private final int batches;
  private final double tolerance;
  private boolean testFeedback = true;
  private boolean testLearning = true;
  private boolean verbose = true;
  private boolean verify = true;
  
  /**
   * Instantiates a new Derivative tester.
   *
   * @param tolerance the tolerance
   * @param probeSize the probe size
   * @param batches   the batches
   */
  public BatchDerivativeTester(final double tolerance, final double probeSize, final int batches) {
    this.tolerance = tolerance;
    this.probeSize = probeSize;
    this.batches = batches;
  }
  
  @javax.annotation.Nonnull
  private Tensor getFeedbackGradient(@javax.annotation.Nonnull final Layer component, final int inputIndex, @javax.annotation.Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor inputTensor = inputPrototype[inputIndex];
    final int inputDims = inputTensor.dim();
    @javax.annotation.Nonnull final Tensor result = new Tensor(inputDims, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      @javax.annotation.Nonnull final PlaceholderLayer<Tensor> inputKey = new PlaceholderLayer<Tensor>(new Tensor());
      @javax.annotation.Nonnull final Result copyInput = new Result(TensorArray.create(inputPrototype), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
        @javax.annotation.Nonnull final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.dim());
        if (!Arrays.equals(inputTensor.getDimensions(), data.get(inputIndex).getDimensions())) {
          throw new AssertionError();
        }
        for (int i = 0; i < inputDims; i++) {
          gradientBuffer.set(new int[]{i, j_}, data.get(inputIndex).getData()[i]);
        }
        buffer.get(inputKey, new double[gradientBuffer.dim()]).addInPlace(gradientBuffer.getData());
      }) {
        
        @Override
        public boolean isAlive() {
          return true;
        }
  
      };
      @javax.annotation.Nullable final Result eval = component.eval(copyInput);
      @javax.annotation.Nonnull final DeltaSet<Layer> xxx = new DeltaSet<Layer>();
      @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(eval.getData().stream().map(x -> {
        @Nonnull Tensor set = x.set(j_, 1);
        x.freeRef();
        return set;
      }).toArray(i -> new Tensor[i]));
      eval.accumulate(xxx, tensorArray);
      tensorArray.freeRef();
      final Delta<Layer> inputDelta = xxx.getMap().get(inputKey);
      if (null != inputDelta) {
        result.addInPlace(new Tensor(inputDelta.getDelta(), result.getDimensions()));
      }
    }
    return result;
  }
  
  @javax.annotation.Nonnull
  private Tensor getLearningGradient(@javax.annotation.Nonnull final Layer component, final int layerNum, @javax.annotation.Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    component.setFrozen(false);
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    @javax.annotation.Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      @javax.annotation.Nonnull final DeltaSet<Layer> buffer = new DeltaSet<Layer>();
      @javax.annotation.Nonnull final Tensor data = new Tensor(outputPrototype.getDimensions()).set((k) -> k == j_ ? 1 : 0);
      @javax.annotation.Nullable final Result eval = component.eval(ConstantResult.singleResultArray(new Tensor[][]{inputPrototype}));
      eval.getData().get(0);
      @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(data);
      eval.accumulate(buffer, tensorArray);
      tensorArray.freeRef();
      final DoubleBuffer<Layer> deltaFlushBuffer = buffer.getMap().values().stream().filter(x -> x.target == stateArray).findFirst().orElse(null);
      if (null != deltaFlushBuffer) {
        for (int i = 0; i < stateLen; i++) {
          gradient.set(new int[]{i, j_}, deltaFlushBuffer.getDelta()[i]);
        }
      }
    }
    return gradient;
  }
  
  /**
   * Is run feedback boolean.
   *
   * @return the boolean
   */
  public boolean isTestFeedback() {
    return testFeedback;
  }
  
  /**
   * Sets run feedback.
   *
   * @param testFeedback the run feedback
   * @return the run feedback
   */
  @javax.annotation.Nonnull
  public BatchDerivativeTester setTestFeedback(final boolean testFeedback) {
    this.testFeedback = testFeedback;
    return this;
  }
  
  /**
   * Is run learning boolean.
   *
   * @return the boolean
   */
  public boolean isTestLearning() {
    return testLearning;
  }
  
  /**
   * Sets run learning.
   *
   * @param testLearning the run learning
   * @return the run learning
   */
  @javax.annotation.Nonnull
  public BatchDerivativeTester setTestLearning(final boolean testLearning) {
    this.testLearning = testLearning;
    return this;
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
  @javax.annotation.Nonnull
  public BatchDerivativeTester setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  /**
   * Is verify boolean.
   *
   * @return the boolean
   */
  public boolean isVerify() {
    return verify;
  }
  
  /**
   * Sets verify.
   *
   * @param verify the verify
   * @return the verify
   */
  @javax.annotation.Nonnull
  public BatchDerivativeTester setVerify(final boolean verify) {
    this.verify = verify;
    return this;
  }
  
  @javax.annotation.Nonnull
  private Tensor measureFeedbackGradient(@javax.annotation.Nonnull final Layer component, final int inputIndex, @javax.annotation.Nonnull final Tensor outputPrototype, @javax.annotation.Nonnull final Tensor... inputPrototype) {
    @javax.annotation.Nonnull final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    @javax.annotation.Nullable final Tensor baseOutput = component.eval(ConstantResult.singleResultArray(new Tensor[][]{inputPrototype})).getData().get(0);
    outputPrototype.set(baseOutput);
    for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
      @javax.annotation.Nonnull final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, probeSize * 1);
      @javax.annotation.Nonnull final Tensor[] copyInput = Arrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      @javax.annotation.Nullable final Tensor evalProbe = component.eval(ConstantResult.singleResultArray(new Tensor[][]{copyInput})).getData().get(0);
      @javax.annotation.Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      for (int j = 0; j < delta.dim(); j++) {
        measuredGradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }
  
  @javax.annotation.Nonnull
  private Tensor measureLearningGradient(@javax.annotation.Nonnull final Layer component, final int layerNum, @javax.annotation.Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    @javax.annotation.Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
  
    @javax.annotation.Nullable final Tensor baseOutput = component.eval(ConstantResult.singleResultArray(new Tensor[][]{inputPrototype})).getData().get(0);
    
    for (int i = 0; i < stateLen; i++) {
      @Nonnull final Layer copy = component.copy();
      copy.state().get(layerNum)[i] += probeSize;
  
      @javax.annotation.Nullable final Tensor evalProbe = copy.eval(ConstantResult.singleResultArray(new Tensor[][]{inputPrototype})).getData().get(0);
      
      @javax.annotation.Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      for (int j = 0; j < delta.dim(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    return gradient;
  }
  
  /**
   * Test learning tolerance statistics.
   *
   * @param component  the component
   * @param IOPair     the io pair
   * @param statistics the statistics
   * @return the tolerance statistics
   */
  public ToleranceStatistics testLearning(@javax.annotation.Nonnull Layer component, @javax.annotation.Nonnull IOPair IOPair, ToleranceStatistics statistics) {
    final ToleranceStatistics prev = statistics;
    statistics = IntStream.range(0, component.state().size()).mapToObj(i -> {
      @Nullable final Tensor measuredGradient = !verify ? null : measureLearningGradient(component, i, IOPair.getOutputPrototype(), IOPair.getInputPrototype());
      @javax.annotation.Nonnull final Tensor implementedGradient = getLearningGradient(component, i, IOPair.getOutputPrototype(), IOPair.getInputPrototype());
      try {
        final ToleranceStatistics result = IntStream.range(0, null == measuredGradient ? 0 : measuredGradient.dim()).mapToObj(i1 -> {
          return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1], implementedGradient.getData()[i1]);
        }).reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());
        if (!(result.absoluteTol.getMax() < tolerance)) {
          throw new AssertionError(result.toString());
        }
        else {
          //log.info(String.format("Component: %s", component));
          if (verbose) {
  
            log.info(String.format("Learning Gradient for weight setByCoord %s", i));
            log.info(String.format("Weights: %s", new Tensor(component.state().get(i)).prettyPrint()));
            log.info(String.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
            log.info(String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
            if (null != measuredGradient) {
              log.info(String.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
              log.info(String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
              log.info(String.format("Gradient Error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
              log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(measuredGradient.minus(implementedGradient).getData())));
            }
          }
          return result;
        }
      } catch (@javax.annotation.Nonnull final Throwable e) {
        //log.info(String.format("Component: %s", component));
        log.info(String.format("Learning Gradient for weight setByCoord %s", i));
        log.info(String.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
        log.info(String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
        if (null != measuredGradient) {
          log.info(String.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
          log.info(String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
          log.info(String.format("Gradient Error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
          log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(measuredGradient.minus(implementedGradient).getData())));
        }
        throw e;
      }
      
    }).reduce((a, b) -> a.combine(b)).map(x -> x.combine(prev)).orElseGet(() -> prev);
    return statistics;
  }
  
  /**
   * Test feedback tolerance statistics.
   *
   * @param component  the component
   * @param IOPair     the io pair
   * @param statistics the statistics
   * @return the tolerance statistics
   */
  public ToleranceStatistics testFeedback(@javax.annotation.Nonnull Layer component, @javax.annotation.Nonnull IOPair IOPair, ToleranceStatistics statistics) {
    statistics = statistics.combine(IntStream.range(0, IOPair.getInputPrototype().length).mapToObj(i -> {
      @Nullable final Tensor measuredGradient = !verify ? null : measureFeedbackGradient(component, i, IOPair.getOutputPrototype(), IOPair.getInputPrototype());
      @javax.annotation.Nonnull final Tensor implementedGradient = getFeedbackGradient(component, i, IOPair.getOutputPrototype(), IOPair.getInputPrototype());
      try {
        final ToleranceStatistics result = IntStream.range(0, null == measuredGradient ? 0 : measuredGradient.dim()).mapToObj(i1 -> {
          return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1], implementedGradient.getData()[i1]);
        }).reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());
        
        if (!(result.absoluteTol.getMax() < tolerance)) throw new AssertionError(result.toString());
        //log.info(String.format("Component: %s", component));
        if (verbose) {
          log.info(String.format("Feedback for input %s", i));
          log.info(String.format("Inputs Values: %s", IOPair.getInputPrototype()[i].prettyPrint()));
          log.info(String.format("Value Statistics: %s", new ScalarStatistics().add(IOPair.getInputPrototype()[i].getData())));
          log.info(String.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
          log.info(String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
          if (null != measuredGradient) {
            log.info(String.format("Measured Feedback: %s", measuredGradient.prettyPrint()));
            log.info(String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
            log.info(String.format("Feedback Error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
            log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(measuredGradient.minus(implementedGradient).getData())));
          }
        }
        return result;
      } catch (@javax.annotation.Nonnull final Throwable e) {
        //log.info(String.format("Component: %s", component));
        log.info(String.format("Feedback for input %s", i));
        log.info(String.format("Inputs Values: %s", IOPair.getInputPrototype()[i].prettyPrint()));
        log.info(String.format("Value Statistics: %s", new ScalarStatistics().add(IOPair.getInputPrototype()[i].getData())));
        log.info(String.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
        log.info(String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
        if (null != measuredGradient) {
          log.info(String.format("Measured: %s", measuredGradient.prettyPrint()));
          log.info(String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
          log.info(String.format("Feedback Error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
          log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(measuredGradient.minus(implementedGradient).getData())));
        }
        throw e;
      }
    }).reduce((a, b) -> a.combine(b)).get());
    return statistics;
  }
  
  /**
   * Test tolerance statistics.
   *
   * @param log
   * @param component      the component
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Override
  public ToleranceStatistics test(@javax.annotation.Nonnull final NotebookOutput log, @javax.annotation.Nonnull final Layer component, @javax.annotation.Nonnull final Tensor... inputPrototype) {
    log.h1("Differential Validation");
    @javax.annotation.Nonnull IOPair ioPair = new IOPair(component, inputPrototype[0]).invoke();
  
    if (verbose) {
      log.code(() -> {
        BatchDerivativeTester.log.info(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get()));
        BatchDerivativeTester.log.info(String.format("Inputs Statistics: %s", Arrays.stream(inputPrototype).map(x -> new ScalarStatistics().add(x.getData()).toString()).reduce((a, b) -> a + ",\n" + b).get()));
        BatchDerivativeTester.log.info(String.format("Output: %s", ioPair.getOutputPrototype().prettyPrint()));
        BatchDerivativeTester.log.info(String.format("Outputs Statistics: %s", new ScalarStatistics().add(ioPair.getOutputPrototype().getData())));
      });
    }
  
    ToleranceStatistics _statistics = new ToleranceStatistics();
  
    if (isTestFeedback()) {
      log.h2("Feedback Validation");
      log.p("We validate the agreement between the implemented derivative _of the inputs_ with finite difference estimations:");
      ToleranceStatistics statistics = _statistics;
      _statistics = log.code(() -> {
        return testFeedback(component, ioPair, statistics);
      });
    }
    if (isTestLearning()) {
      log.h2("Learning Validation");
      log.p("We validate the agreement between the implemented derivative _of the internal weights_ with finite difference estimations:");
      ToleranceStatistics statistics = _statistics;
      _statistics = log.code(() -> {
        return testLearning(component, ioPair, statistics);
      });
    }
  
    log.h2("Total Accuracy");
    log.p("The overall agreement accuracy between the implemented derivative and the finite difference estimations:");
    ToleranceStatistics statistics = _statistics;
    log.code(() -> {
      //log.info(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toString(inputPrototype), outputPrototype));
      BatchDerivativeTester.log.info(String.format("Finite-Difference Derivative Accuracy:"));
      BatchDerivativeTester.log.info(String.format("absoluteTol: %s", statistics.absoluteTol));
      BatchDerivativeTester.log.info(String.format("relativeTol: %s", statistics.relativeTol));
    });
  
    log.h2("Frozen and Alive Status");
    log.code(() -> {
      testFrozen(component, ioPair.getInputPrototype());
      testUnFrozen(component, ioPair.getInputPrototype());
    });
  
    return _statistics;
  }
  
  /**
   * Test frozen.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public void testFrozen(@javax.annotation.Nonnull final Layer component, @javax.annotation.Nonnull final Tensor[] inputPrototype) {
    @javax.annotation.Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().freeze();
    @javax.annotation.Nullable final Result eval = frozen.eval(new Result(TensorArray.create(inputPrototype), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      reachedInputFeedback.set(true);
    }) {
  
      @Override
      public boolean isAlive() {
        return true;
      }
  
  
    });
    @javax.annotation.Nonnull final DeltaSet<Layer> buffer = new DeltaSet<Layer>();
    TensorList tensorList = eval.getData().copy();
    eval.accumulate(buffer, tensorList);
    tensorList.freeRef();
    final List<Delta<Layer>> deltas = component.state().stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(Collectors.toList());
    if (!deltas.isEmpty() && !component.state().isEmpty()) {
      throw new AssertionError("Frozen component listed in delta. Deltas: " + deltas);
    }
    final int inElements = Arrays.stream(inputPrototype).mapToInt(x -> x.dim()).sum();
    if (!reachedInputFeedback.get() && 0 < inElements) {
      throw new RuntimeException("Frozen component did not pass input backwards");
    }
  }
  
  /**
   * Test un frozen.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public void testUnFrozen(@javax.annotation.Nonnull final Layer component, final Tensor[] inputPrototype) {
    @javax.annotation.Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().setFrozen(false);
    @javax.annotation.Nullable final Result eval = frozen.eval(new Result(TensorArray.create(inputPrototype), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      reachedInputFeedback.set(true);
    }) {
  
      @Override
      public boolean isAlive() {
        return true;
      }
  
    });
    @javax.annotation.Nonnull final DeltaSet<Layer> buffer = new DeltaSet<Layer>();
    TensorList data = eval.getData();
    eval.accumulate(buffer, data);
    data.freeRef();
    @Nullable final List<double[]> stateList = frozen.state();
    final List<Delta<Layer>> deltas = stateList.stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(Collectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      throw new AssertionError("Nonfrozen component not listed in delta. Deltas: " + deltas);
    }
    if (!reachedInputFeedback.get()) {
      throw new RuntimeException("Nonfrozen component did not pass input backwards");
    }
  }
  
  @javax.annotation.Nonnull
  @Override
  public String toString() {
    return "BatchDerivativeTester{" +
      "probeSize=" + probeSize +
      ", batches=" + batches +
      ", tolerance=" + tolerance +
      ", testFeedback=" + testFeedback +
      ", testLearning=" + testLearning +
      ", verbose=" + verbose +
      ", verify=" + verify +
      '}';
  }
  
  private class IOPair {
    private final Layer component;
    private final Tensor tensor;
    private Tensor[] inputPrototype;
    private Tensor outputPrototype;
  
    /**
     * Instantiates a new Io pair.
     *
     * @param component the component
     * @param tensor    the tensor
     */
    public IOPair(Layer component, Tensor tensor) {
      this.component = component;
      this.tensor = tensor;
    }
  
    /**
     * Get input prototype tensor [ ].
     *
     * @return the tensor [ ]
     */
    public Tensor[] getInputPrototype() {
      return inputPrototype;
    }
  
    /**
     * Gets output prototype.
     *
     * @return the output prototype
     */
    public Tensor getOutputPrototype() {
      return outputPrototype;
    }
  
    /**
     * Invoke io pair.
     *
     * @return the io pair
     */
    @javax.annotation.Nonnull
    public IOPair invoke() {
      inputPrototype = IntStream.range(0, batches).mapToObj(i -> tensor.copy()).toArray(j -> new Tensor[j]);
      outputPrototype = SimpleEval.run(component, inputPrototype[0]).getOutputAndFree();
      return this;
    }
  }
}
