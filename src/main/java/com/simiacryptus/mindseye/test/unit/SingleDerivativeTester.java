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

import com.simiacryptus.mindseye.lang.ConstantResult;
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.DoubleBuffer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
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
import java.util.Optional;
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
  
  @Nonnull
  private Tensor getFeedbackGradient(@Nonnull final Layer component, final int inputIndex, @Nonnull final Tensor outputPrototype, @Nonnull final Tensor... inputPrototype) {
    final Tensor inputTensor = inputPrototype[inputIndex];
    final int inputDims = inputTensor.length();
    @Nonnull final Tensor result = new Tensor(inputDims, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final PlaceholderLayer<Tensor> inputKey = new PlaceholderLayer<Tensor>(new Tensor(1));
      inputKey.getKey().freeRef();
      final Result[] copyInput = Arrays.stream(inputPrototype).map(x -> new Result(TensorArray.create(x), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {}) {
        
        @Override
        public boolean isAlive() {
          return false;
        }
    
      }).toArray(i -> new Result[i]);
      copyInput[inputIndex].getData().freeRef();
      copyInput[inputIndex].freeRef();
      double[] target = new double[inputDims * outputPrototype.length()];
      copyInput[inputIndex] = new Result(TensorArray.create(inputTensor), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
        if (1 != data.length()) throw new AssertionError();
        if (data.length() != 1) throw new AssertionError();
        @Nonnull final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.length());
        if (!Arrays.equals(inputTensor.getDimensions(), data.getDimensions())) {
          throw new AssertionError();
        }
        IntStream.range(0, data.length()).forEach(dataIndex -> {
          for (int i = 0; i < inputDims; i++) {
            @Nullable Tensor tensor = data.get(dataIndex);
            gradientBuffer.set(new int[]{i, j_}, tensor.getData()[i]);
            tensor.freeRef();
          }
        });
        buffer.get(inputKey, target).addInPlace(gradientBuffer.getData()).freeRef();
        gradientBuffer.freeRef();
      }) {
        
        @Override
        public boolean isAlive() {
          return true;
        }
      };
      @Nullable final Result eval;
      try {
        eval = component.eval(copyInput);
      } finally {
        for (@Nonnull Result nnResult : copyInput) {
          nnResult.freeRef();
          nnResult.getData().freeRef();
        }
      }
      @Nonnull final DeltaSet<Layer> deltaSet = new DeltaSet<Layer>();
      @Nonnull TensorArray tensorArray = TensorArray.wrap(new Tensor(outputPrototype.getDimensions()).set(j, 1));
      try {
        eval.accumulate(deltaSet, tensorArray);
      } finally {
        eval.getData().freeRef();
        eval.freeRef();
      }
      final Delta<Layer> inputDelta = deltaSet.getMap().get(inputKey);
      if (null != inputDelta) {
        @Nonnull Tensor tensor = new Tensor(inputDelta.getDelta(), result.getDimensions());
        result.addInPlace(tensor);
        tensor.freeRef();
      }
      deltaSet.freeRef();
      inputKey.freeRef();
    }
    return result;
  }
  
  @Nonnull
  private Tensor getLearningGradient(@Nonnull final Layer component, final int layerNum, @Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    component.setFrozen(false);
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());
    for (int j = 0; j < outputPrototype.length(); j++) {
      final int j_ = j;
      @Nonnull final DeltaSet<Layer> buffer = new DeltaSet<Layer>();
      Result[] array = ConstantResult.batchResultArray(new Tensor[][]{inputPrototype});
      @Nullable final Result eval = component.eval(array);
      for (@Nonnull Result result : array) {
        result.getData().freeRef();
        result.freeRef();
      }
      @Nonnull TensorArray tensorArray = TensorArray.wrap(new Tensor(outputPrototype.getDimensions()).set((k) -> k == j_ ? 1 : 0));
      eval.accumulate(buffer, tensorArray);
      eval.getData().freeRef();
      eval.freeRef();
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
   * Is apply feedback boolean.
   *
   * @return the boolean
   */
  public boolean isTestFeedback() {
    return testFeedback;
  }
  
  /**
   * Sets apply feedback.
   *
   * @param testFeedback the apply feedback
   * @return the apply feedback
   */
  @Nonnull
  public SingleDerivativeTester setTestFeedback(final boolean testFeedback) {
    this.testFeedback = testFeedback;
    return this;
  }
  
  /**
   * Is apply learning boolean.
   *
   * @return the boolean
   */
  public boolean isTestLearning() {
    return testLearning;
  }
  
  /**
   * Sets apply learning.
   *
   * @param testLearning the apply learning
   * @return the apply learning
   */
  @Nonnull
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
  @Nonnull
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
  @Nonnull
  public SingleDerivativeTester setVerify(final boolean verify) {
    this.verify = verify;
    return this;
  }
  
  @Nonnull
  private Tensor measureFeedbackGradient(@Nonnull final Layer component, final int inputIndex, @Nonnull final Tensor outputPrototype, @Nonnull final Tensor... inputPrototype) {
    @Nonnull final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].length(), outputPrototype.length());
    Result[] input0 = ConstantResult.batchResultArray(new Tensor[][]{inputPrototype});
    @Nullable final Tensor baseOutput = component.eval(input0).getDataAndFree().getAndFree(0);
    for (@Nonnull Result result : input0) {
      result.freeRef();
      result.getData().freeRef();
    }
    outputPrototype.set(baseOutput);
    for (int i = 0; i < inputPrototype[inputIndex].length(); i++) {
      @Nonnull final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, probeSize * 1);
      @Nonnull final Tensor[] copyInput = Arrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      Result[] input1 = ConstantResult.batchResultArray(new Tensor[][]{copyInput});
      try {
        @Nullable final Tensor evalProbe = component.eval(input1).getDataAndFree().getAndFree(0);
        @Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
        for (int j = 0; j < delta.length(); j++) {
          measuredGradient.set(new int[]{i, j}, delta.getData()[j]);
        }
        evalProbe.freeRef();
        delta.freeRef();
      } finally {
        inputProbe.freeRef();
        for (@Nonnull Result result : input1) {
          result.freeRef();
          result.getData().freeRef();
        }
        
      }
    }
    baseOutput.freeRef();
    return measuredGradient;
  }
  
  @Nonnull
  private Tensor measureLearningGradient(@Nonnull final Layer component, final int layerNum, @Nonnull final Tensor outputPrototype, final Tensor... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    @Nonnull final Tensor gradient = new Tensor(stateLen, outputPrototype.length());
    
    Result[] input2 = ConstantResult.batchResultArray(new Tensor[][]{inputPrototype});
    @Nullable final Tensor baseOutput = component.eval(input2).getDataAndFree().getAndFree(0);
    
    for (int i = 0; i < stateLen; i++) {
      @Nonnull final Layer copy = component.copy();
      copy.state().get(layerNum)[i] += probeSize;
      @Nullable final Tensor evalProbe = copy.eval(input2).getDataAndFree().getAndFree(0);
      copy.freeRef();
      @Nonnull final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      evalProbe.freeRef();
      for (int j = 0; j < delta.length(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
      delta.freeRef();
    }
    baseOutput.freeRef();
    for (@Nonnull Result result : input2) {
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
  public ToleranceStatistics test(@Nonnull final NotebookOutput output, @Nonnull final Layer component, @Nonnull final Tensor... inputPrototype) {
    output.h1("Differential Validation");
    ToleranceStatistics _statistics = new ToleranceStatistics();
    final Tensor outputPrototype = SimpleEval.run(component, inputPrototype).getOutputAndFree();
  
    if (verbose) {
      output.code(() -> {
        log.info(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).orElse("")));
        log.info(String.format("Inputs Statistics: %s", Arrays.stream(inputPrototype).map(x -> new ScalarStatistics().add(x.getData()).toString()).reduce((a, b) -> a + ",\n" + b).orElse("")));
        log.info(String.format("Output: %s", null == outputPrototype ? null : outputPrototype.prettyPrint()));
        log.info(String.format("Outputs Statistics: %s", new ScalarStatistics().add(outputPrototype.getData())));
      });
    }
    if (isTestFeedback()) {
      output.h2("Feedback Validation");
      output.p("We validate the agreement between the implemented derivative _of the inputs_ apply finite difference estimations:");
      final ToleranceStatistics statistics = _statistics;
      _statistics = output.code(() -> {
        return testFeedback(statistics, component, inputPrototype, outputPrototype);
      });
    }
    if (isTestLearning()) {
      output.h2("Learning Validation");
      output.p("We validate the agreement between the implemented derivative _of the internal weights_ apply finite difference estimations:");
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
  public ToleranceStatistics testLearning(@Nonnull ToleranceStatistics prev, @Nonnull Layer component, Tensor[] inputPrototype, @Nonnull Tensor outputPrototype) {
    return IntStream.range(0, component.state().size()).mapToObj(i -> {
      @Nullable final Tensor measuredGradient = !verify ? null : measureLearningGradient(component, i, outputPrototype, inputPrototype);
      @Nonnull final Tensor implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
      @Nonnull Tensor difference = measuredGradient.minus(implementedGradient);
      try {
        final ToleranceStatistics result = IntStream.range(0, null == measuredGradient ? 0 : measuredGradient.length()).mapToObj(i1 -> {
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
      } catch (@Nonnull final Throwable e) {
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
  public ToleranceStatistics testFeedback(@Nonnull ToleranceStatistics statistics, @Nonnull Layer component, @Nonnull Tensor[] inputPrototype, @Nonnull Tensor outputPrototype) {
    Optional<ToleranceStatistics> optional = IntStream.range(0, inputPrototype.length).mapToObj(i -> {
      @Nullable final Tensor measuredGradient = !verify ? null : measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
      @Nonnull final Tensor implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype);
      @Nonnull Tensor difference = measuredGradient.minus(implementedGradient);
      try {
        final ToleranceStatistics result = IntStream.range(0, null == measuredGradient ? 0 : measuredGradient.length()).mapToObj(i1 -> {
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
      } catch (@Nonnull final Throwable e) {
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
    }).reduce((a, b) -> a.combine(b));
    if (!optional.isPresent()) return statistics;
    return statistics.combine(optional.orElse(null));
  }
  
  /**
   * Test frozen.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public void testFrozen(@Nonnull final Layer component, @Nonnull Tensor[] inputPrototype) {
    final int inElements = Arrays.stream(inputPrototype).mapToInt(x -> x.length()).sum();
    inputPrototype = Arrays.stream(inputPrototype).map(tensor -> tensor.copy()).toArray(i -> new Tensor[i]);
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().freeze();
    List<TensorArray> inputCopies = Arrays.stream(inputPrototype).map(TensorArray::wrap).collect(Collectors.toList());
    Result[] input = inputCopies.stream().map((tensorArray) -> new Result(tensorArray, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
      reachedInputFeedback.set(true);
    }) {
    
      @Override
      public boolean isAlive() {
        return true;
      }
    
    }).toArray(i -> new Result[i]);
    @Nullable final Result eval;
    try {
      eval = frozen.eval(input);
    } finally {
      for (@Nonnull Result result : input) {
        result.freeRef();
      }
      frozen.freeRef();
      for (@Nonnull TensorArray tensorArray : inputCopies) {
        tensorArray.freeRef();
      }
    }
    @Nonnull final DeltaSet<Layer> buffer;
    TensorList tensorList;
    TensorList evalData = eval.getData();
    try {
      buffer = new DeltaSet<Layer>();
      tensorList = evalData.copy();
      eval.accumulate(buffer, tensorList);
    } finally {
      evalData.freeRef();
      eval.freeRef();
    }
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
  public void testUnFrozen(@Nonnull final Layer component, Tensor[] inputPrototype) {
    inputPrototype = Arrays.stream(inputPrototype).map(tensor -> tensor.copy()).toArray(i -> new Tensor[i]);
    @Nonnull final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    @Nonnull final Layer frozen = component.copy().setFrozen(false);
    List<TensorArray> inputCopies = Arrays.stream(inputPrototype).map(TensorArray::wrap).collect(Collectors.toList());
    Result[] inputs = inputCopies.stream().map(tensor -> new Result(tensor, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
      reachedInputFeedback.set(true);
    }) {
    
      @Override
      public boolean isAlive() {
        return true;
      }
    
    }).toArray(i -> new Result[i]);
    @Nullable final Result eval = frozen.eval(inputs);
    for (@Nonnull Result result : inputs) {
      result.freeRef();
    }
    for (@Nonnull TensorArray tensorArray : inputCopies) {
      tensorArray.freeRef();
    }
    @Nonnull final DeltaSet<Layer> buffer = new DeltaSet<Layer>();
    TensorList tensorList = eval.getData();
    eval.accumulate(buffer, tensorList);
    eval.freeRef();
    @Nullable final List<double[]> stateList = frozen.state();
    final List<Delta<Layer>> deltas = stateList.stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(Collectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      throw new AssertionError("Nonfrozen component not listed in delta. Deltas: " + deltas);
    }
    frozen.freeRef();
    buffer.freeRef();
    if (!reachedInputFeedback.get() && inputPrototype.length != 0) {
      throw new RuntimeException("Nonfrozen component did not pass input backwards");
    }
  }
  
  @Nonnull
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
