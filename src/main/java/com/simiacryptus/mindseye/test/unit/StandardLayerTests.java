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
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.CodeUtil;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.function.Consumer;

/**
 * The type Layer test base.
 */
public abstract class StandardLayerTests {
  /**
   * The constant originalOut.
   */
  public static final PrintStream originalOut = System.out;
  private static final Logger log = LoggerFactory.getLogger(StandardLayerTests.class);
  private ArrayList<ComponentTest> bigTests;
  private ArrayList<ComponentTest> littleTests;
  /**
   * The Validate batch execution.
   */
  protected boolean validateBatchExecution = true;
  /**
   * The Validate differentials.
   */
  protected boolean validateDifferentials = true;
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void test(NotebookOutput log) {
    if (null != originalOut) log.addCopy(originalOut);
    NNLayer layer = getLayer(getInputDims());
    log.h1("%s", layer.getClass().getSimpleName());
    log.p(String.format("Layer Type %s", log.link(CodeUtil.findFile(layer.getClass()), layer.getClass().getCanonicalName())));
    log.p(CodeUtil.getJavadoc(layer.getClass()));
    log.h2("%s", getClass().getSimpleName());
    log.p(String.format("Test Type %s", log.link(CodeUtil.findFile(getClass()), getClass().getCanonicalName())));
    log.p(CodeUtil.getJavadoc(getClass()));
    if (layer instanceof DAGNetwork) {
      log.h3("Network Diagram");
      log.p("This is a network with the following layout:");
      log.code(() -> {
        return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
          .height(400).width(600).render(Format.PNG).toImage();
      });
    }
    getLittleTests().stream().filter(x -> null != x).forEach(test -> {
      test.test(log, layer.copy(), randomize(getInputDims()));
    });
    NNLayer perfLayer = getLayer(getPerfDims());
    getBigTests().stream().filter(x -> null != x).forEach(test -> {
      test.test(log, perfLayer.copy(), randomize(getPerfDims()));
    });
  }
  
  /**
   * Random tensor [ ].
   *
   * @param inputDims the input dims
   * @return the tensor [ ]
   */
  public Tensor[] randomize(int[][] inputDims) {
    return Arrays.stream(inputDims).map(dim -> new Tensor(dim).fill(() -> random())).toArray(i -> new Tensor[i]);
  }
  
  /**
   * Gets batching tester.
   *
   * @return the batching tester
   */
  public ComponentTest getBatchingTester() {
    if (!validateBatchExecution) return null;
    return new BatchingTester(1e-2) {
      @Override
      public double getRandom() {
        return random();
      }
    };
  }
  
  /**
   * Random double.
   *
   * @return the double
   */
  public double random() {
    return Math.round(1000.0 * (Util.R.get().nextDouble() - 0.5)) / 250.0;
  }
  
  /**
   * Gets equivalency tester.
   *
   * @return the equivalency tester
   */
  public ComponentTest getEquivalencyTester() {
    NNLayer referenceLayer = getReferenceLayer();
    if (null == referenceLayer) return null;
    return new EquivalencyTester(1e-2, referenceLayer);
  }
  
  /**
   * Gets performance tester.
   *
   * @return the performance tester
   */
  public PerformanceTester getPerformanceTester() {
    return new PerformanceTester();
  }
  
  /**
   * Gets reference io tester.
   *
   * @return the reference io tester
   */
  protected ComponentTest getReferenceIOTester() {
    return new ReferenceIO(getReferenceIO());
  }
  
  /**
   * Gets json tester.
   *
   * @return the json tester
   */
  protected ComponentTest getJsonTester() {
    return new JsonTest();
  }
  
  /**
   * Gets reference io.
   *
   * @return the reference io
   */
  protected HashMap<Tensor[], Tensor> getReferenceIO() {
    return new HashMap<>();
  }
  
  /**
   * Gets derivative tester.
   *
   * @return the derivative tester
   */
  public ComponentTest getDerivativeTester() {
    if (!validateDifferentials) return null;
    return new SingleDerivativeTester(1e-3, 1e-4);
  }
  
  /**
   * Gets layer.
   *
   * @param inputSize the input size
   * @return the layer
   */
  public abstract NNLayer getLayer(int[][] inputSize);
  
  /**
   * Gets reference layer.
   *
   * @return the reference layer
   */
  public NNLayer getReferenceLayer() {
    return null;
  }
  
  /**
   * Get input dims int [ ] [ ].
   *
   * @return the int [ ] [ ]
   */
  public abstract int[][] getInputDims();
  
  /**
   * Get perf dims int [ ] [ ].
   *
   * @return the int [ ] [ ]
   */
  public int[][] getPerfDims() {
    return getInputDims();
  }
  
  /**
   * Gets learning tester.
   *
   * @return the learning tester
   */
  public ComponentTest getTrainingTester() {
    return new TrainingTester();
  }
  
  /**
   * Gets little tests.
   *
   * @return the little tests
   */
  public ArrayList<ComponentTest> getLittleTests() {
    if(null == littleTests) {
      synchronized (this) {
        if(null == littleTests) {
          littleTests = new ArrayList<>(Arrays.asList(
            getJsonTester(), getReferenceIOTester(), getBatchingTester(), getDerivativeTester(), getEquivalencyTester()
          ));
        }
      }
    }
    return littleTests;
  }
  
  /**
   * With little tests standard layer tests.
   *
   * @param fn the fn
   * @return the standard layer tests
   */
  public StandardLayerTests withLittleTests(Consumer<ArrayList<ComponentTest>> fn) {
    fn.accept(getLittleTests());
    return this;
  }
  
  /**
   * With big tests standard layer tests.
   *
   * @param fn the fn
   * @return the standard layer tests
   */
  public StandardLayerTests withBigTests(Consumer<ArrayList<ComponentTest>> fn) {
    fn.accept(getBigTests());
    return this;
  }
  
  /**
   * Gets big tests.
   *
   * @return the big tests
   */
  public ArrayList<ComponentTest> getBigTests() {
    if(null == bigTests) {
      synchronized (this) {
        if(null == bigTests) {
          bigTests = new ArrayList<>(Arrays.asList(
            getPerformanceTester(), getTrainingTester()
          ));
        }
      }
    }
    return bigTests;
  }
  
}
