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
public class SingleDerivativeTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger log = LoggerFactory.getLogger(SingleDerivativeTester.class);
  
  /**
   * The Probe size.
   */
  public final double probeSize;
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
   */
  public SingleDerivativeTester(final double tolerance, final double probeSize) {
    this.tolerance = tolerance;
    this.probeSize = probeSize;
  }
  
  @javax.annotation.Nonnull
  private Tensor getFeedbackGradient(@javax.annotation.Nonnull final Layer component, final int inputIndex, @javax.annotation.Nonnull final Tensor outputPrototype, @javax.annotation.Nonnull final Tensor... inputPrototype) {
    final Tensor inputTensor = inputPrototype[inputIndex];
    final int inputDims = inputTensor.dim();
    @javax.annotation.Nonnull final Tensor result = new Tensor(inputDims, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      @javax.annotation.Nonnull final PlaceholderLayer<Tensor> inputKey = new PlaceholderLayer<Tensor>(new Tensor());
      inputKey.getKey().freeRef();
      final Result[] copyInput = Arrays.stream(inputPrototype).map(x -> new Result(TensorArray.create(x), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {}) {
        
        @Override
        public boolean isAlive() {
          return false;
        }
    
      }).toArray(i -> new Result[i]);
      copyInput[inputIndex].getData().freeRef();
      copyInput[inputIndex].freeRef();
      copyInput[inputIndex] = new Result(TensorArray.create(inputTensor), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
        if (1 != data.length()) throw new AssertionError();
        if (data.length() != 1) throw new AssertionError();
        @javax.annotation.Nonnull final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.dim());
        if (!Arrays.equals(inputTensor.getDimensions(), data.getDimensions())) {
          throw new AssertionError();
        }
        IntStream.range(0, data.length()).forEach(dataIndex -> {
          for (int i = 0; i < inputDims; i++) {
            @javax.annotation.Nullable Tensor tensor = data.get(dataIndex);
            gradientBuffer.set(new int[]{i, j_}, tensor.getData()[i]);
            tensor.freeRef();
          }
        });
        buffer.get(inputKey, new double[gradientBuffer.dim()]).addInPlace(gradientBuffer.getData()).freeRef();
        gradientBuffer.freeRef();
      }) {
        
        @Override
        public boolean isAlive() {
          return true;
        }
      };
      @javax.annotation.Nullable final Result eval = component.eval(copyInput);
      for (@javax.annotation.Nonnull Result nnResult : copyInput) {
        nnResult.freeRef();
        nnResult.getData().freeRef();
      }
      @javax.annotation.Nonnull final DeltaSet<Layer> deltaSet = new DeltaSet<Layer>();
      @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(new Tensor(outputPrototype.getDimensions()).set(j, 1));
      eval.accumulate(deltaSet, tensorArray);
      eval.getData().freeRef();
      eval.freeRef();
      tensorArray.freeRef();
      final Delta<Layer> inputDelta = deltaSet.getMap().get(inputKey);
      if (null != inputDelta) {
        @javax.annotation.Nonnull Tensor tensor = new Tensor(inputDelta.getDelta(), result.getDimensions());
        result.addInPlace(tensor);
        tensor.freeRef();
      }
      deltaSet.freeRef();
      inputKey.freeRef();
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
      Result[] array = ConstantResult.batchResultArray(new Tensor[][]{inputPrototype});
      @javax.annotation.Nullable final Result eval = component.eval(array);
      for (@javax.annotation.Nonnull Result result : array) {
        result.getData().freeRef();
        result.freeRef();
      }
      @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(new Tensor(outputPrototype.getDimensions()).set((k) -> k == j_ ? 1 : 0));
      eval.accumulate(buffer, tensorArray);
      eval.getData().freeRef();
      eval.freeRef();
      tensorArray.freeRef();
      final DoubleBuffer<Layer> deltaFlushBuffer = buffer.getMap().values().stream().filter(x -> x.target == stateArray).findFirst().orElse(null);
      if (null != deltaFlushBuffer) {
        for (int i = 0; i < stateLen; i++) {
          gradient.set(new int[]{i, j_}, deltaFlushBuffer.getDelta()[i]);
        }
      }
      buffer.freeRef();
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
  public SingleDerivativeTester setTestFeedback(final boolean testFeedback) {
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
  public SingleDerivativeTester setTestLearning(final boolean testLearning) {
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
  public SingleDerivativeTester setVerbose(final boolean verbose) {
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
  public SingleDerivativeTester setVerify(final boolean verify) {
    this.verify = verify;
    return this;
  }
  
  @javax.annotation.Nonnull
  private Tensor measureFeedbackGradient(@javax.annotation.Nonnull final Layer component, final int inputIndex, @javax.annotation.Nonnull final Tensor outputPrototype, @javax.annotation.Nonnull final Tensor... inputPrototype) {
    @javax.annotation.Nonnull final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    Result[] input0 = ConstantResult.batchResultArray(new Tensor[][]{inputPrototype});
    @javax.annotation.Nullable final Tensor baseOutput = component.eval(input0).getDataAndFree().getAndFree(0);
    for (@javax.annotation.Nonnull Result result : input0) {
      result.freeRef();
      result.getData().freeRef();
    }
    outputPrototype.set(baseOutput);
    for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
      @javax.annotation.Nonnull final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, probeSize * 1);
      @javax.annotation.Nonnull final Tensor[] copyInput = Arrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      Result[] input1 = ConstantResult.batchResultArray(new Tensor[][]{copyInput});
      @javax.annotation.Nullable final Tensor evalProbe = component.eval(input1).getDataAndFree().getAndFree(0);
      inputProbe.freeRef();
      for (@javax.annotation.Nonnull Result result : input1) {
        result.freeRef();
        result.getData().freeRef();
      }
      @javax.annotation.Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      evalProbe.freeRef();
      for (int j = 0; j < delta.dim(); j++) {
        measuredGradient.set(new int[]{i, j}, delta.getData()[j]);
      }
      delta.freeRef();
    }
    baseOutput.freeRef();
    return measuredGradient;
  }
  
  @javax.annotation.Nonnull
  private Tensor measureLearningGradient(@javax.annotation.Nonnull final Layer component, final int layerNum, @javax.annotation.Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    @javax.annotation.Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
  
    Result[] input2 = ConstantResult.batchResultArray(new Tensor[][]{inputPrototype});
    @javax.annotation.Nullable final Tensor baseOutput = component.eval(input2).getDataAndFree().getAndFree(0);
    
    for (int i = 0; i < stateLen; i++) {
      @Nonnull final Layer copy = component.copy();
      copy.state().get(layerNum)[i] += probeSize;
      @javax.annotation.Nullable final Tensor evalProbe = copy.eval(input2).getDataAndFree().getAndFree(0);
      copy.freeRef();
      @javax.annotation.Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      evalProbe.freeRef();
      for (int j = 0; j < delta.dim(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
      delta.freeRef();
    }
    baseOutput.freeRef();
    for (@javax.annotation.Nonnull Result result : input2) {
      result.freeRef();
      result.getData().freeRef();
    }
    return gradient;
  }
  
  /**
   * Test tolerance statistics.
   *
   * @param output
   * @param component      the component
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Override
  public ToleranceStatistics test(@javax.annotation.Nonnull final NotebookOutput output, @javax.annotation.Nonnull final Layer component, @javax.annotation.Nonnull final Tensor... inputPrototype) {
    output.h1("Differential Validation");
    ToleranceStatistics _statistics = new ToleranceStatistics();
    final Tensor outputPrototype = SimpleEval.run(component, inputPrototype).getOutputAndFree();
  
    if (verbose) {
      output.code(() -> {
        log.info(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get()));
        log.info(String.format("Inputs Statistics: %s", Arrays.stream(inputPrototype).map(x -> new ScalarStatistics().add(x.getData()).toString()).reduce((a, b) -> a + ",\n" + b).get()));
        log.info(String.format("Output: %s", outputPrototype.prettyPrint()));
        log.info(String.format("Outputs Statistics: %s", new ScalarStatistics().add(outputPrototype.getData())));
      });
    }
    if (isTestFeedback()) {
      output.h2("Feedback Validation");
      output.p("We validate the agreement between the implemented derivative _of the inputs_ with finite difference estimations:");
      final ToleranceStatistics statistics = _statistics;
      _statistics = output.code(() -> {
        return testFeedback(statistics, component, inputPrototype, outputPrototype);
      });
    }
    if (isTestLearning()) {
      output.h2("Learning Validation");
      output.p("We validate the agreement between the implemented derivative _of the internal weights_ with finite difference estimations:");
      final ToleranceStatistics statistics = _statistics;
      _statistics = output.code(() -> {
        return testLearning(statistics, component, inputPrototype, outputPrototype);
      });
    }
  
    outputPrototype.freeRef();
    output.h2("Total Accuracy");
    output.p("The overall agreement accuracy between the implemented derivative and the finite difference estimations:");
    final ToleranceStatistics statistics = _statistics;
    output.code(() -> {
      //log.info(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toString(inputPrototype), outputPrototype));
      log.info(String.format("Finite-Difference Derivative Accuracy:"));
      log.info(String.format("absoluteTol: %s", statistics.absoluteTol));
      log.info(String.format("relativeTol: %s", statistics.relativeTol));
    });
  
    output.h2("Frozen and Alive Status");
    output.code(() -> {
      testFrozen(component, inputPrototype);
      testUnFrozen(component, inputPrototype);
    });
  
    return _statistics;
  }
  
  /**
   * Test learning tolerance statistics.
   *
   * @param prev            the prev
   * @param component       the component
   * @param inputPrototype  the input prototype
   * @param outputPrototype the output prototype
   * @return the tolerance statistics
   */
  public ToleranceStatistics testLearning(@Nonnull ToleranceStatistics prev, @javax.annotation.Nonnull Layer component, Tensor[] inputPrototype, @javax.annotation.Nonnull Tensor outputPrototype) {
    return IntStream.range(0, component.state().size()).mapToObj(i -> {
      @Nullable final Tensor measuredGradient = !verify ? null : measureLearningGradient(component, i, outputPrototype, inputPrototype);
      @javax.annotation.Nonnull final Tensor implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
      @javax.annotation.Nonnull Tensor difference = measuredGradient.minus(implementedGradient);
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
            log.info(String.format("Weights: %s", Tensor.prettyPrint(component.state().get(i))));
            log.info(String.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
            log.info(String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
            if (null != measuredGradient) {
              log.info(String.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
              log.info(String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
              log.info(String.format("Gradient Error: %s", difference.prettyPrint()));
              log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
            }
          }
          difference.freeRef();
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
          log.info(String.format("Gradient Error: %s", difference.prettyPrint()));
          log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
        }
        difference.freeRef();
        throw e;
      } finally {
        measuredGradient.freeRef();
        implementedGradient.freeRef();
      }
      
    }).reduce((a, b) -> a.combine(b)).map(x -> x.combine(prev)).orElseGet(() -> prev);
  }
  
  /**
   * Test feedback tolerance statistics.
   *
   * @param statistics      the statistics
   * @param component       the component
   * @param inputPrototype  the input prototype
   * @param outputPrototype the output prototype
   * @return the tolerance statistics
   */
  @Nonnull
  public ToleranceStatistics testFeedback(@javax.annotation.Nonnull ToleranceStatistics statistics, @javax.annotation.Nonnull Layer component, @javax.annotation.Nonnull Tensor[] inputPrototype, @javax.annotation.Nonnull Tensor outputPrototype) {
    return statistics.combine(IntStream.range(0, inputPrototype.length).mapToObj(i -> {
      @Nullable final Tensor measuredGradient = !verify ? null : measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
      @javax.annotation.Nonnull final Tensor implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype);
      @javax.annotation.Nonnull Tensor difference = measuredGradient.minus(implementedGradient);
      try {
        final ToleranceStatistics result = IntStream.range(0, null == measuredGradient ? 0 : measuredGradient.dim()).mapToObj(i1 -> {
          return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1], implementedGradient.getData()[i1]);
        }).reduce((a, b) -> a.combine(b)).orElse(new ToleranceStatistics());
        
        if (!(result.absoluteTol.getMax() < tolerance)) throw new AssertionError(result.toString());
        //log.info(String.format("Component: %s", component));
        if (verbose) {
          log.info(String.format("Feedback for input %s", i));
          log.info(String.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
          log.info(String.format("Value Statistics: %s", new ScalarStatistics().add(inputPrototype[i].getData())));
          log.info(String.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
          log.info(String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
          if (null != measuredGradient) {
            log.info(String.format("Measured Feedback: %s", measuredGradient.prettyPrint()));
            log.info(String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
            log.info(String.format("Feedback Error: %s", difference.prettyPrint()));
            log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
          }
        }
        difference.freeRef();
        measuredGradient.freeRef();
        implementedGradient.freeRef();
        return result;
      } catch (@javax.annotation.Nonnull final Throwable e) {
        //log.info(String.format("Component: %s", component));
        log.info(String.format("Feedback for input %s", i));
        log.info(String.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
        log.info(String.format("Value Statistics: %s", new ScalarStatistics().add(inputPrototype[i].getData())));
        log.info(String.format("Implemented Feedback: %s", implementedGradient.prettyPrint()));
        log.info(String.format("Implemented Statistics: %s", new ScalarStatistics().add(implementedGradient.getData())));
        if (null != measuredGradient) {
          log.info(String.format("Measured: %s", measuredGradient.prettyPrint()));
          log.info(String.format("Measured Statistics: %s", new ScalarStatistics().add(measuredGradient.getData())));
          log.info(String.format("Feedback Error: %s", difference.prettyPrint()));
          log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(difference.getData())));
        }
        measuredGradient.freeRef();
        implementedGradient.freeRef();
        difference.freeRef();
        throw e;
      }
    }).reduce((a, b) -> a.combine(b)).get());
  }
  
  /**
   * Test frozen.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public void testFrozen(@javax.annotation.Nonnull final Layer component, @javax.annotation.Nonnull Tensor[] inputPrototype) {
    final int inElements = Arrays.stream(inputPrototype).mapToInt(x -> x.dim()).sum();
    inputPrototype = Arrays.stream(inputPrototype).map(tensor -> tensor.copy()).toArray(i -> new Tensor[i]);
    @javax.annotation.Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().freeze();
    List<TensorArray> inputCopies = Arrays.stream(inputPrototype).map(TensorArray::wrap).collect(Collectors.toList());
    Result[] input = inputCopies.stream().map((tensorArray) -> new Result(tensorArray, (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      reachedInputFeedback.set(true);
    }) {
  
      @Override
      public boolean isAlive() {
        return true;
      }
    
    }).toArray(i -> new Result[i]);
    @javax.annotation.Nullable final Result eval = frozen.eval(input);
    for (@javax.annotation.Nonnull Result result : input) {
      result.freeRef();
    }
    frozen.freeRef();
    for (@javax.annotation.Nonnull TensorArray tensorArray : inputCopies) {
      tensorArray.freeRef();
    }
    @javax.annotation.Nonnull final DeltaSet<Layer> buffer = new DeltaSet<Layer>();
    TensorList tensorList = eval.getData().copy();
    eval.accumulate(buffer, tensorList);
    eval.getData().freeRef();
    eval.freeRef();
    tensorList.freeRef();
    final List<Delta<Layer>> deltas = component.state().stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(Collectors.toList());
    buffer.freeRef();
    if (!deltas.isEmpty() && !component.state().isEmpty()) {
      throw new AssertionError("Frozen component listed in delta. Deltas: " + deltas);
    }
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
  public void testUnFrozen(@javax.annotation.Nonnull final Layer component, Tensor[] inputPrototype) {
    inputPrototype = Arrays.stream(inputPrototype).map(tensor -> tensor.copy()).toArray(i -> new Tensor[i]);
    @javax.annotation.Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().setFrozen(false);
    List<TensorArray> inputCopies = Arrays.stream(inputPrototype).map(TensorArray::wrap).collect(Collectors.toList());
    Result[] inputs = inputCopies.stream().map(tensor -> new Result(tensor, (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      reachedInputFeedback.set(true);
    }) {
  
      @Override
      public boolean isAlive() {
        return true;
      }
    
    }).toArray(i -> new Result[i]);
    @javax.annotation.Nullable final Result eval = frozen.eval(inputs);
    for (@javax.annotation.Nonnull Result result : inputs) {
      result.freeRef();
    }
    for (@javax.annotation.Nonnull TensorArray tensorArray : inputCopies) {
      tensorArray.freeRef();
    }
    @javax.annotation.Nonnull final DeltaSet<Layer> buffer = new DeltaSet<Layer>();
    TensorList tensorList = eval.getData();
    eval.accumulate(buffer, tensorList);
    eval.freeRef();
    tensorList.freeRef();
    @Nullable final List<double[]> stateList = frozen.state();
    final List<Delta<Layer>> deltas = stateList.stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(Collectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      throw new AssertionError("Nonfrozen component not listed in delta. Deltas: " + deltas);
    }
    frozen.freeRef();
    buffer.freeRef();
    if (!reachedInputFeedback.get()) {
      throw new RuntimeException("Nonfrozen component did not pass input backwards");
    }
  }
  
  @javax.annotation.Nonnull
  @Override
  public String toString() {
    return "SingleDerivativeTester{" +
      "probeSize=" + probeSize +
      ", tolerance=" + tolerance +
      ", testFeedback=" + testFeedback +
      ", testLearning=" + testLearning +
      ", verbose=" + verbose +
      ", verify=" + verify +
      '}';
  }
}
