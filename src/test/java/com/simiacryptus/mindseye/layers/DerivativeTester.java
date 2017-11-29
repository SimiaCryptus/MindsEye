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

package com.simiacryptus.mindseye.layers;

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.SimpleEval;
import com.simiacryptus.util.io.KryoUtil;
import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * The type Derivative tester.
 */
public class DerivativeTester {
  
  private static final Logger log = LoggerFactory.getLogger(DerivativeTester.class);
  
  /**
   * The Probe size.
   */
  public final double probeSize;
  private final double tolerance;
  private boolean testLearning = true;
  private boolean testFeedback = true;
  
  /**
   * Instantiates a new Derivative tester.
   *
   * @param tolerance the tolerance
   * @param probeSize the probe size
   */
  public DerivativeTester(double tolerance, double probeSize) {
    this.tolerance = tolerance;
    this.probeSize = probeSize;
  }
  
  /**
   * Test.
   *
   * @param component      the component
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  public ToleranceStatistics test(final NNLayer component, final Tensor... inputPrototype) {
    Tensor outputPrototype = SimpleEval.run(component, inputPrototype).getOutput();
    ToleranceStatistics statistics = new ToleranceStatistics();
    if (isTestFeedback()) {
      statistics = statistics.combine(IntStream.range(0, inputPrototype.length).mapToObj(i -> {
        return testFeedback(component, i, outputPrototype, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).get());
    }
    if (isTestLearning()) {
      ToleranceStatistics prev = statistics;
      statistics = IntStream.range(0, component.state().size()).mapToObj(i -> {
        return testLearning(component, i, outputPrototype, inputPrototype);
      }).reduce((a, b) -> a.combine(b)).map(x->x.combine(prev)).orElseGet(()->prev);
    }
    //System.out.println(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.toString(inputPrototype), outputPrototype));
    System.out.println(String.format("Finite-Difference Derivative Accuracy:"));
    System.out.println(String.format("absoluteTol: %s", statistics.absoluteTol));
    System.out.println(String.format("relativeTol: %s", statistics.relativeTol));
    
    //testFrozen(component, inputPrototype);
  
    return statistics;
  }
  
  public void testFrozen(NNLayer component, Tensor[] inputPrototype) {
    AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    NNLayer frozen = component.copy().freeze();
    GpuController.run(exe->{
      NNResult eval = frozen.eval(exe, Arrays.stream(inputPrototype).map((Tensor x) -> new NNResult(x) {
        @Override
        public void accumulate(DeltaSet buffer, TensorList data) {
          reachedInputFeedback.set(true);
        }
        
        @Override
        public boolean isAlive() {
          return true;
        }
      }).<NNResult>toArray(i -> new NNResult[i]));
      DeltaSet buffer = new DeltaSet();
      eval.accumulate(buffer, eval.getData());
      if (buffer.getMap().containsKey(component)) throw new AssertionError("Frozen component listed in delta");
      if(!reachedInputFeedback.get()) throw new RuntimeException("Frozen component did not pass input backwards");
    });
  }
  
  public void testUnFrozen(NNLayer component, Tensor[] inputPrototype) {
    AtomicBoolean reachedInputFeedback = new AtomicBoolean(false);
    NNLayer frozen = component.copy().freeze();
    GpuController.run(exe->{
      NNResult eval = frozen.eval(exe, Arrays.stream(inputPrototype).map((Tensor x) -> new NNResult(x) {
        @Override
        public void accumulate(DeltaSet buffer, TensorList data) {
          reachedInputFeedback.set(true);
        }
        
        @Override
        public boolean isAlive() {
          return true;
        }
      }).<NNResult>toArray(i -> new NNResult[i]));
      DeltaSet buffer = new DeltaSet();
      eval.accumulate(buffer, eval.getData());
      if (!buffer.getMap().containsKey(component)) throw new AssertionError("Frozen component not listed in delta");
      if(!reachedInputFeedback.get()) throw new RuntimeException("Non-Frozen component did not pass input backwards");
    });
  }
  
  private Tensor getFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor gradientBuffer = new Tensor(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final NNResult[] copyInput = Arrays.stream(inputPrototype).map(x -> new NNResult(x) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList data) {
        }
        
        @Override
        public boolean isAlive() {
          return false;
        }
      }).toArray(i -> new NNResult[i]);
      copyInput[inputIndex] = new NNResult(inputPrototype[inputIndex]) {
        @Override
        public void accumulate(final DeltaSet buffer, final TensorList data) {
          Assert.assertEquals(1, data.length());
          IntStream.range(0, data.length()).forEach(dataIndex -> {
            Assert.assertArrayEquals(inputPrototype[inputIndex].getDimensions(), data.get(dataIndex).getDimensions());
            for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
              gradientBuffer.set(new int[]{i, j_}, data.get(dataIndex).getData()[i]);
            }
          });
        }
        
        @Override
        public boolean isAlive() {
          return true;
        }
      };
      final Tensor[] data = {new Tensor(outputPrototype.getDimensions()).fill((k) -> k == j_ ? 1 : 0)};
      GpuController.INSTANCE.distribute(Arrays.<Tensor[]>asList(inputPrototype),
        (d, exe) -> {
          NNResult eval = component.eval(exe, copyInput);
          Tensor tensor = eval.getData().get(0);
          eval.accumulate(new DeltaSet(), new TensorArray(data));
          return tensor;
        }, (a, b) -> a.add(b));
    }
    return gradientBuffer;
  }
  
  private Tensor getLearningGradient(final NNLayer component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
    component.setFrozen(false);
    final double[] stateArray = component.state().get(layerNum);
    final int stateLen = stateArray.length;
    final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
    for (int j = 0; j < outputPrototype.dim(); j++) {
      final int j_ = j;
      final DeltaSet buffer = new DeltaSet();
      final Tensor[] data = {new Tensor(outputPrototype.getDimensions()).fill((k) -> k == j_ ? 1 : 0)};
      GpuController.call(exe->{
        NNResult eval = component.eval(exe, NNResult.batchResultArray(new Tensor[][]{inputPrototype}));
        Tensor tensor = eval.getData().get(0);
        eval.accumulate(buffer, new TensorArray(data));
        return tensor;
      });
      final DoubleBuffer deltaFlushBuffer = buffer.getMap().values().stream().filter(x -> x.target == stateArray).findFirst().get();
      for (int i = 0; i < stateLen; i++) {
        gradient.set(new int[]{i, j_}, deltaFlushBuffer.getDelta()[i]);
      }
    }
    return gradient;
  }
  
  private Tensor measureFeedbackGradient(final NNLayer component, final int inputIndex, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor measuredGradient = new Tensor(inputPrototype[inputIndex].dim(), outputPrototype.dim());
    Tensor baseOutput = GpuController.call(exe->{
      return component.eval(exe, NNResult.batchResultArray(new Tensor[][]{inputPrototype})).getData().get(0);
    });
    outputPrototype.set(baseOutput);
    for (int i = 0; i < inputPrototype[inputIndex].dim(); i++) {
      final Tensor inputProbe = inputPrototype[inputIndex].copy();
      inputProbe.add(i, probeSize * 1);
      final Tensor[] copyInput = Arrays.copyOf(inputPrototype, inputPrototype.length);
      copyInput[inputIndex] = inputProbe;
      Tensor evalProbe = GpuController.call(exe->{
        return component.eval(exe, NNResult.batchResultArray(new Tensor[][]{copyInput})).getData().get(0);
      });
      final Tensor delta = evalProbe.minus(baseOutput).scale(1. / probeSize);
      for (int j = 0; j < delta.dim(); j++) {
        measuredGradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    return measuredGradient;
  }
  
  private Tensor measureLearningGradient(final NNLayer component, final int layerNum, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final int stateLen = component.state().get(layerNum).length;
    final Tensor gradient = new Tensor(stateLen, outputPrototype.dim());
  
    Tensor baseOutput = GpuController.call(exe -> {
      return component.eval(exe, NNResult.batchResultArray(new Tensor[][]{inputPrototype})).getData().get(0);
    });
    
    for (int i = 0; i < stateLen; i++) {
      final NNLayer copy = KryoUtil.kryo().copy(component);
      copy.state().get(layerNum)[i] += probeSize;
      
      Tensor evalProbe = GpuController.call(exe->{
        return copy.eval(exe, NNResult.batchResultArray(new Tensor[][]{inputPrototype})).getData().get(0);
      });
      
      final Tensor delta = evalProbe.minus(baseOutput).scale(1. / probeSize);
      for (int j = 0; j < delta.dim(); j++) {
        gradient.set(new int[]{i, j}, delta.getData()[j]);
      }
    }
    return gradient;
  }
  
  /**
   * Test feedback.
   *
   * @param component       the component
   * @param i               the
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @return the tolerance statistics
   */
  protected ToleranceStatistics testFeedback(final NNLayer component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor measuredGradient = measureFeedbackGradient(component, i, outputPrototype, inputPrototype);
    final Tensor implementedGradient = getFeedbackGradient(component, i, outputPrototype, inputPrototype);
    try {
      ToleranceStatistics result = IntStream.range(0, measuredGradient.dim()).mapToObj(i1 -> {
        return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1], implementedGradient.getData()[i1]);
      }).reduce((a, b) -> a.combine(b)).get();
  
      if (!(result.absoluteTol.getMax() < tolerance)) throw new AssertionError(result.toString());
      System.out.println(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(), outputPrototype.prettyPrint()));
      System.out.println(String.format("measured/actual: %s", measuredGradient.prettyPrint()));
      System.out.println(String.format("implemented/expected: %s", implementedGradient.prettyPrint()));
      System.out.println(String.format("error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
      return result;
    } catch (final Throwable e) {
      System.out.println(String.format("Component: %s\nInputs: %s\noutput=%s", component, Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(), outputPrototype.prettyPrint()));
      System.out.println(String.format("measured/actual: %s", measuredGradient.prettyPrint()));
      System.out.println(String.format("implemented/expected: %s", implementedGradient.prettyPrint()));
      System.out.println(String.format("error: %s", measuredGradient.minus(implementedGradient).prettyPrint()));
      throw e;
    }
  }
  
  /**
   * Test learning.
   *
   * @param component       the component
   * @param i               the
   * @param outputPrototype the output prototype
   * @param inputPrototype  the input prototype
   * @return the tolerance statistics
   */
  protected ToleranceStatistics testLearning(final NNLayer component, final int i, final Tensor outputPrototype, final Tensor... inputPrototype) {
    final Tensor measuredGradient = measureLearningGradient(component, i, outputPrototype, inputPrototype);
    final Tensor implementedGradient = getLearningGradient(component, i, outputPrototype, inputPrototype);
    final Tensor error = measuredGradient.minus(implementedGradient);
  
    try {
      ToleranceStatistics result = IntStream.range(0, measuredGradient.dim()).mapToObj(i1 -> {
        return new ToleranceStatistics().accumulate(measuredGradient.getData()[i1], implementedGradient.getData()[i1]);
      }).reduce((a, b) -> a.combine(b)).get();
      if (!(result.absoluteTol.getMax() < tolerance)) {
        throw new AssertionError(result.toString());
      } else {
        System.out.println(String.format("Component: %s", component));
        System.out.println(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get()));
        System.out.println(String.format("Outputs: %s", outputPrototype.prettyPrint()));
        System.out.println(String.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
        System.out.println(String.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
        System.out.println(String.format("Error: %s", error.prettyPrint()));
        return result;
      }
    } catch (final Throwable e) {
      System.out.println(String.format("Component: %s", component));
      System.out.println(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get()));
      System.out.println(String.format("Outputs: %s", outputPrototype.prettyPrint()));
      System.out.println(String.format("Measured Gradient: %s", measuredGradient.prettyPrint()));
      System.out.println(String.format("Implemented Gradient: %s", implementedGradient.prettyPrint()));
      System.out.println(String.format("Error: %s", error.prettyPrint()));
      throw e;
    }
    
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
  public DerivativeTester setTestLearning(boolean testLearning) {
    this.testLearning = testLearning;
    return this;
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
  public DerivativeTester setTestFeedback(boolean testFeedback) {
    this.testFeedback = testFeedback;
    return this;
  }
}
