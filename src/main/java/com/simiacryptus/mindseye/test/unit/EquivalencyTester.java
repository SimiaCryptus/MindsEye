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

import com.google.gson.GsonBuilder;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The type Equivalency tester.
 */
public class EquivalencyTester extends ComponentTestBase<ToleranceStatistics> {
  private static final Logger log = LoggerFactory.getLogger(EquivalencyTester.class);

  private final Layer reference;
  private final double tolerance;

  /**
   * Instantiates a new Equivalency tester.
   *
   * @param tolerance      the tolerance
   * @param referenceLayer the reference key
   */
  public EquivalencyTester(final double tolerance, final Layer referenceLayer) {
    this.tolerance = tolerance;
    this.reference = referenceLayer;
    this.reference.addRef();
  }

  @Override
  protected void _free() {
    reference.freeRef();
    super._free();
  }

  /**
   * Test tolerance statistics.
   *
   * @param subject        the subject
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  public ToleranceStatistics test(@Nullable final Layer subject, @Nonnull final Tensor[] inputPrototype) {
    if (null == reference || null == subject) return new ToleranceStatistics();
    reference.assertAlive();
    final Tensor subjectOutput = SimpleEval.run(subject, inputPrototype).getOutputAndFree();
    final Tensor referenceOutput = SimpleEval.run(reference, inputPrototype).getOutputAndFree();
    @Nonnull final Tensor error = subjectOutput.minus(referenceOutput);
    @Nonnull final ToleranceStatistics result = IntStream.range(0, subjectOutput.length()).mapToObj(i1 -> {
      return new ToleranceStatistics().accumulate(subjectOutput.getData()[i1], referenceOutput.getData()[i1]);
    }).reduce((a, b) -> a.combine(b)).get();
    try {
      try {
        if (!(result.absoluteTol.getMax() < tolerance)) throw new AssertionError(result.toString());
      } catch (@Nonnull final Throwable e) {
        log.info(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b)));
        log.info(String.format("Subject Output: %s", subjectOutput.prettyPrint()));
        log.info(String.format("Reference Output: %s", referenceOutput.prettyPrint()));
        log.info(String.format("Error: %s", error.prettyPrint()));
        System.out.flush();
        throw e;
      }
      log.info(String.format("Inputs: %s", Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get()));
      log.info(String.format("Error: %s", error.prettyPrint()));
      log.info(String.format("Accuracy:"));
      log.info(String.format("absoluteTol: %s", result.absoluteTol.toString()));
      log.info(String.format("relativeTol: %s", result.relativeTol.toString()));
      return result;
    } finally {
      subjectOutput.freeRef();
      referenceOutput.freeRef();
      error.freeRef();
    }
  }

  /**
   * Test tolerance statistics.
   *
   * @param output
   * @param subject        the subject
   * @param inputPrototype the input prototype
   * @return the tolerance statistics
   */
  @Override
  public ToleranceStatistics test(@Nonnull final NotebookOutput output, final Layer subject, @Nonnull final Tensor... inputPrototype) {
    output.h1("Reference Implementation");
    output.p("This key is an alternate implementation which is expected to behave the same as the following key:");
    output.run(() -> {
      log.info(new GsonBuilder().setPrettyPrinting().create().toJson(reference.getJson()));
    });
    output.p("We measureStyle the agreement between the two layers in a random execution:");
    return output.eval(() -> {
      return test(subject, inputPrototype);
    });
  }

  @Nonnull
  @Override
  public String toString() {
    return "EquivalencyTester{" +
        "reference=" + reference +
        ", tolerance=" + tolerance +
        '}';
  }
}
