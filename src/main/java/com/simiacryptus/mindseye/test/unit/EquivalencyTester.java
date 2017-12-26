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

import com.google.gson.GsonBuilder;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * The type Equivalency tester.
 */
public class EquivalencyTester implements ComponentTest<ToleranceStatistics> {
  private static final Logger logger = LoggerFactory.getLogger(EquivalencyTester.class);
  
  private final NNLayer reference;
  private final double tolerance;
  
  /**
   * Instantiates a new Equivalency tester.
   *
   * @param tolerance      the tolerance
   * @param referenceLayer the reference layer
   */
  public EquivalencyTester(final double tolerance, final NNLayer referenceLayer) {
    this.tolerance = tolerance;
    reference = referenceLayer;
  }
  
  /**
   * Test tolerance statistics.
   *
   * @param subject        the subject
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  public ToleranceStatistics test(final NNLayer subject, final Tensor[] inputPrototype) {
    if (null == reference || null == subject) return new ToleranceStatistics();
    ToleranceStatistics result1;
    final Tensor subjectOutput = SimpleEval.run(subject, inputPrototype).getOutput();
    final Tensor referenceOutput = SimpleEval.run(reference, inputPrototype).getOutput();
    final Tensor error = subjectOutput.minus(referenceOutput);
    final ToleranceStatistics result = IntStream.range(0, subjectOutput.dim()).mapToObj(i1 -> {
      return new ToleranceStatistics().accumulate(subjectOutput.getData()[i1], referenceOutput.getData()[i1]);
    }).reduce((a, b) -> a.combine(b)).get();
    try {
      if (!(result.absoluteTol.getMax() < tolerance)) throw new AssertionError(result.toString());
      result1 = result;
    } catch (final Throwable e) {
      logger.info(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b)));
      logger.info(String.format("Subject Output: %s", subjectOutput.prettyPrint()));
      logger.info(String.format("Reference Output: %s", referenceOutput.prettyPrint()));
      logger.info(String.format("Error: %s", error.prettyPrint()));
      System.out.flush();
      throw e;
    }
    final ToleranceStatistics statistics = result1;
    logger.info(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get()));
    logger.info(String.format("Error: %s", error.prettyPrint()));
    logger.info(String.format("Accuracy:"));
    logger.info(String.format("absoluteTol: %s", statistics.absoluteTol.toString()));
    logger.info(String.format("relativeTol: %s", statistics.relativeTol.toString()));
    return statistics;
  }
  
  /**
   * Test tolerance statistics.
   *
   * @param log
   * @param subject        the subject
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Override
  public ToleranceStatistics test(final NotebookOutput log, final NNLayer subject, final Tensor... inputPrototype) {
    log.h1("Reference Implementation");
    log.p("This layer is an alternate implementation which is expected to behave the same as the following layer:");
    log.code(() -> {
      logger.info(new GsonBuilder().setPrettyPrinting().create().toJson(reference.getJson()));
    });
    log.p("We measure the agreement between the two layers in a random execution:");
    return log.code(() -> {
      return test(subject, inputPrototype);
    });
  }
  
}
