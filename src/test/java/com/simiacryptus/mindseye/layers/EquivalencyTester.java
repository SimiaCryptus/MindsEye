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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * The type Derivative tester.
 */
public class EquivalencyTester {
  
  private static final Logger log = LoggerFactory.getLogger(EquivalencyTester.class);
  
  /**
   * The Probe size.
   */
  private final double tolerance;
  
  /**
   * Instantiates a new Derivative tester.
   *
   * @param tolerance the tolerance
   */
  public EquivalencyTester(double tolerance) {
    this.tolerance = tolerance;
  }
  
  /**
   * Test.
   *
   * @param reference      the component
   * @param subject        the subject
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  public ToleranceStatistics test(final NNLayer reference, final NNLayer subject, final Tensor... inputPrototype) {
    if(null == reference || null == subject) return new ToleranceStatistics();
    ToleranceStatistics result1;
    final Tensor subjectOutput = SimpleEval.run(subject, inputPrototype).getOutput();
    final Tensor referenceOutput = SimpleEval.run(reference, inputPrototype).getOutput();
    Tensor error = subjectOutput.minus(referenceOutput);
    ToleranceStatistics result = IntStream.range(0, subjectOutput.dim()).mapToObj(i1 -> {
      return new ToleranceStatistics().accumulate(subjectOutput.getData()[i1], referenceOutput.getData()[i1]);
    }).reduce((a, b) -> a.combine(b)).get();
    try {
      if (!(result.absoluteTol.getMax() < tolerance)) throw new AssertionError(result.toString());
      result1 = result;
    } catch (Throwable e) {
      System.out.println(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a, b)->a+",\n"+b)));
      System.out.println(String.format("Subject Output: %s", subjectOutput.prettyPrint()));
      System.out.println(String.format("Reference Output: %s", referenceOutput.prettyPrint()));
      System.out.println(String.format("Error: %s", error.prettyPrint()));
      System.out.flush();
      throw e;
    }
    ToleranceStatistics statistics = result1;
    System.out.println(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a, b)->a+",\n"+b).get()));
    System.out.println(String.format("Error: %s", error.prettyPrint()));
    System.out.println(String.format("Accuracy:"));
    System.out.println(String.format("absoluteTol: %s", statistics.absoluteTol.toString()));
    System.out.println(String.format("relativeTol: %s", statistics.relativeTol.toString()));
    return statistics;
  }
  
}
