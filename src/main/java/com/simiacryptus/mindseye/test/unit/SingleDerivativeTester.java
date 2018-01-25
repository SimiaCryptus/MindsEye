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
import com.simiacryptus.util.io.KryoUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Derivative tester.
 */
public class SingleDerivativeTester implements ComponentTest<ToleranceStatistics> {
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
  
  private Tensor getFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor inputTensor = inputPrototype[inputIndex];
    final int inputDims = inputTensor.dim();
    final Tensor result = new Tensor(inputDims, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final PlaceholderLayer<Tensor> inputKey = new PlaceholderLayer<Tensor>(new Tensor());
      final NNResult[] copyInput = Arrays.stream(inputPrototype).map(x -> new NNResult(TensorArray.create(x), (final DeltaSet<NNLayer> buffer, final TensorList data) -> {}) {
        
        @Override
        public boolean isAlive() {
          return false;
        }
  
      }).toArray(i -> new NNResult[i]);
      copyInput[inputIndex] = new NNResult(TensorArray.create(inputTensor), (final DeltaSet<NNLayer> buffer, final TensorList data) -> {
        if (1 != data.length()) throw new AssertionError();
        if (data.length() != 1) throw new AssertionError();
        final Tensor gradientBuffer = new Tensor(inputDims, outputPrototype.dim());
        IntStream.range(0, data.length()).forEach(dataIndex -> {
          if (!Arrays.equals(inputTensor.getDimensions(), data.get(dataIndex).getDimensions())) {
            throw new AssertionError();
          }
          for (int i = 0; i < inputDims; i++) {
            gradientBuffer.set(new int[]{i, j_}, data.get(dataIndex).getData()[i]);
          }
        });
        buffer.get(inputKey, new double[gradientBuffer.dim()]).addInPlace(gradientBuffer.getData());
      }) {
        
        @Override
        public boolean isAlive() {
          return true;
        }
      };
      final NNResult eval = component.eval(copyInput);
      final DeltaSet<NNLayer> deltaSet = new DeltaSet<NNLayer>();
      TensorArray tensorArray = TensorArray.create(new Tensor(outputPrototype.getDimensions()).set(j, 1));
      eval.accumulate(deltaSet, tensorArray);
      tensorArray.freeRef();
      final Delta<NNLayer> inputDelta = deltaSet.getMap().get(inputKey);
      if (null != inputDelta) {
        result.addInPlace(new Tensor(inputDelta.getDelta(), result.getDimensions()));
      }
    }
    return result;
  }
  
  private Tensor getLearningGradient(final NNLayer component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
    component.setFrozen(false);
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final DeltaSet<NNLayer> buffer = new DeltaSet<NNLayer>();
      final NNResult eval = component.eval(NNConstant.batchResultArray(new Tensor[][]{inputPrototype}));
      TensorArray tensorArray = TensorArray.create(new Tensor(outputPrototype.getDimensions()).set((k) -> k == j_ ? 1 : 0));
      eval.accumulate(buffer, tensorArray);
      tensorArray.freeRef();
      final DoubleBuffer<NNLayer> deltaFlushBuffer = buffer.getMap().values().stream().filter(x -> x.target == stateArray).findFirst().orElse(null);
      if (null != deltaFlushBuffer) {
        for (int i = 0; i < stateLen; i++) {
          gradient.set(new int[]{i, j_}, deltaFlushBuffer.getDelta()[i]);
        }
      }
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
  public SingleDerivativeTester setVerify(final boolean verify) {
    this.verify = verify;
    return this;
  }
  
  private Tensor measureFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    final Tensor baseOutput = component.eval(NNConstant.batchResultArray(new Tensor[][]{inputPrototype})).getData().get(0);
    outputPrototype.set(baseOutput);
    for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
      final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, probeSize * 1);
      final Tensor[] copyInput = Arrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      final Tensor evalProbe = component.eval(NNConstant.batchResultArray(new Tensor[][]{copyInput})).getData().get(0);
      final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      for (int j = 0; j < delta.dim(); j++) {
        measuredGradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }
  
  private Tensor measureLearningGradient(final NNLayer component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
  
    final Tensor baseOutput = component.eval(NNConstant.batchResultArray(new Tensor[][]{inputPrototype})).getData().get(0);
    
    for (int i = 0; i < stateLen; i++) {
      final NNLayer copy = KryoUtil.kryo().copy(component);
      copy.state().get(layerNum)[i] += probeSize;
  
      final Tensor evalProbe = copy.eval(NNConstant.batchResultArray(new Tensor[][]{inputPrototype})).getData().get(0);
      
      final Tensor delta = evalProbe.minus(baseOutput).scaleInPlace(1. / probeSize);
      for (int j = 0; j < delta.dim(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
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
  public ToleranceStatistics test(final NotebookOutput output, final NNLayer component, final Tensor... inputPrototype) {
    output.h1("Differential Validation");
    ToleranceStatistics _statistics = new ToleranceStatistics();
    final Tensor outputPrototype = SimpleEval.run(component, inputPrototype).getOutput();
  
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
  public ToleranceStatistics testLearning(ToleranceStatistics prev, NNLayer component, Tensor[] inputPrototype, Tensor outputPrototype) {
    return IntStream.range(0, component.state().size()).mapToObj(i -> {
      final Tensor measuredGradient = !verify ? null : measureLearningGradient(component, i, outputPrototype, inputPrototype);
      final Tensor implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
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
      } catch (final Throwable e) {
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
  public ToleranceStatistics testFeedback(ToleranceStatistics statistics, NNLayer component, Tensor[] inputPrototype, Tensor outputPrototype) {
    return statistics.combine(IntStream.range(0, inputPrototype.length).mapToObj(i -> {
      final Tensor measuredGradient = !verify ? null : measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
      final Tensor implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype);
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
            log.info(String.format("Feedback Error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
            log.info(String.format("Error Statistics: %s", new ScalarStatistics().add(measuredGradient.minus(implementedGradient).getData())));
          }
        }
        return result;
      } catch (final Throwable e) {
        //log.info(String.format("Component: %s", component));
        log.info(String.format("Feedback for input %s", i));
        log.info(String.format("Inputs Values: %s", inputPrototype[i].prettyPrint()));
        log.info(String.format("Value Statistics: %s", new ScalarStatistics().add(inputPrototype[i].getData())));
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
  }
  
  /**
   * Test frozen.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   */
  public void testFrozen(final NNLayer component, Tensor[] inputPrototype) {
    final int inElements = Arrays.stream(inputPrototype).mapToInt(x -> x.dim()).sum();
    inputPrototype = Arrays.stream(inputPrototype).map(tensor -> tensor.copy()).toArray(i -> new Tensor[i]);
    final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    final NNLayer frozen = component.copy().freeze();
    List<TensorArray> inputCopies = Arrays.stream(inputPrototype).map(TensorArray::wrap).collect(Collectors.toList());
    final NNResult eval = frozen.eval(inputCopies.stream().map((tensorArray) -> new NNResult(tensorArray, (final DeltaSet<NNLayer> buffer, final TensorList data) -> {
      reachedInputFeedback.set(true);
    }) {
  
      @Override
      public boolean isAlive() {
        return true;
      }
  
    }).<NNResult>toArray(i -> new NNResult[i]));
    for (TensorArray tensorArray : inputCopies) {
      tensorArray.freeRef();
    }
    final DeltaSet<NNLayer> buffer = new DeltaSet<NNLayer>();
    TensorList tensorList = eval.getData().copy();
    eval.accumulate(buffer, tensorList);
    tensorList.freeRef();
    final List<Delta<NNLayer>> deltas = component.state().stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(Collectors.toList());
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
  public void testUnFrozen(final NNLayer component, Tensor[] inputPrototype) {
    inputPrototype = Arrays.stream(inputPrototype).map(tensor -> tensor.copy()).toArray(i -> new Tensor[i]);
    final AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    final NNLayer frozen = component.copy().setFrozen(false);
    List<TensorArray> inputCopies = Arrays.stream(inputPrototype).map(TensorArray::wrap).collect(Collectors.toList());
    final NNResult eval = frozen.eval(inputCopies.stream().map(tensor -> new NNResult(tensor, (final DeltaSet<NNLayer> buffer, final TensorList data) -> {
      reachedInputFeedback.set(true);
    }) {
  
      @Override
      public boolean isAlive() {
        return true;
      }
  
    }).<NNResult>toArray(i -> new NNResult[i]));
    for (TensorArray tensorArray : inputCopies) {
      tensorArray.freeRef();
    }
    final DeltaSet<NNLayer> buffer = new DeltaSet<NNLayer>();
    TensorList tensorList = eval.getData();
    eval.accumulate(buffer, tensorList);
    tensorList.freeRef();
    final List<double[]> stateList = frozen.state();
    final List<Delta<NNLayer>> deltas = stateList.stream().map(doubles -> {
      return buffer.stream().filter(x -> x.target == doubles).findFirst().orElse(null);
    }).filter(x -> x != null).collect(Collectors.toList());
    if (deltas.isEmpty() && !stateList.isEmpty()) {
      throw new AssertionError("Nonfrozen component not listed in delta. Deltas: " + deltas);
    }
    if (!reachedInputFeedback.get()) {
      throw new RuntimeException("Nonfrozen component did not pass input backwards");
    }
  }
}
