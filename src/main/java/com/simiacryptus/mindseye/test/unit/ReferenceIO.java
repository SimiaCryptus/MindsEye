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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.io.NotebookOutput;

import java.util.Arrays;
import java.util.HashMap;

/**
 * The type Reference io.
 */
public class ReferenceIO implements ComponentTest<ToleranceStatistics> {
  /**
   * The Reference io.
   */
  HashMap<Tensor[], Tensor> referenceIO;
  
  /**
   * Instantiates a new Reference io.
   *
   * @param referenceIO the reference io
   */
  public ReferenceIO(final HashMap<Tensor[], Tensor> referenceIO) {
    this.referenceIO = referenceIO;
  }
  
  @Override
  public ToleranceStatistics test(final NotebookOutput log, final NNLayer layer, final Tensor... inputPrototype) {
    if (!referenceIO.isEmpty()) {
      log.h1("Reference Input/Output Pairs");
      log.p("Display pre-set input/output example pairs:");
      referenceIO.forEach((input, output) -> {
        log.code(() -> {
          final SimpleEval eval = SimpleEval.run(layer, input);
          final DoubleStatistics error = new DoubleStatistics().accept(eval.getOutput().add(output.scale(-1)).getData());
          return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\nError: %s\n--------------------\nDerivative: \n%s",
            Arrays.stream(input).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
            eval.getOutput().prettyPrint(), error,
            Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
        });
      });
    }
    else {
      log.h1("Example Input/Output Pair");
      log.p("Display input/output pairs from random executions:");
      log.code(() -> {
        final SimpleEval eval = SimpleEval.run(layer, inputPrototype);
        return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
          Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
          eval.getOutput().prettyPrint(),
          Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
      });
    }
    return null;
  }
}
