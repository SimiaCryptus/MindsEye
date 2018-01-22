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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.Explodable;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.SysOutInterceptor;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * The type Layer apply base.
 */
public abstract class StandardLayerTests extends NotebookReportBase {
  static {
    SysOutInterceptor.INSTANCE.init();
  }
  
  /**
   * The Validate batch execution.
   */
  protected boolean validateBatchExecution = true;
  /**
   * The Validate differentials.
   */
  protected boolean validateDifferentials = true;
  private ArrayList<ComponentTest<?>> finalTests;
  private ArrayList<ComponentTest<?>> bigTests;
  private ArrayList<ComponentTest<?>> littleTests;
  
  /**
   * The constant seed.
   */
  public static final long seed = 51389; //System.nanoTime();
  private boolean testTraining = true;
  
  /**
   * Instantiates a new Standard layer tests.
   */
  public StandardLayerTests() {
    logger.info("Seed: " + seed);
  }
  
  /**
   * Gets batching tester.
   *
   * @return the batching tester
   */
  public ComponentTest<ToleranceStatistics> getBatchingTester() {
    if (!validateBatchExecution) return null;
    return new BatchingTester(1e-2) {
      @Override
      public double getRandom() {
        return random();
      }
    };
  }
  
  /**
   * Gets big tests.
   *
   * @return the big tests
   */
  public ArrayList<ComponentTest<?>> getBigTests() {
    if (null == bigTests) {
      synchronized (this) {
        if (null == bigTests) {
          bigTests = new ArrayList<>(Arrays.asList(
            getPerformanceTester(),
            getBatchingTester(),
            getReferenceIOTester(),
            getEquivalencyTester()
                                                  ));
        }
      }
    }
    return bigTests;
  }
  
  /**
   * Gets big tests.
   *
   * @return the big tests
   */
  public ArrayList<ComponentTest<?>> getFinalTests() {
    if (null == finalTests) {
      synchronized (this) {
        if (null == finalTests) {
          finalTests = new ArrayList<>(Arrays.asList(
            getTrainingTester()
                                                    ));
        }
      }
    }
    return finalTests;
  }
  
  /**
   * Gets derivative tester.
   *
   * @return the derivative tester
   */
  public ComponentTest<ToleranceStatistics> getDerivativeTester() {
    if (!validateDifferentials) return null;
    return new SingleDerivativeTester(1e-3, 1e-4);
  }
  
  /**
   * Gets equivalency tester.
   *
   * @return the equivalency tester
   */
  public ComponentTest<ToleranceStatistics> getEquivalencyTester() {
    final NNLayer referenceLayer = getReferenceLayer();
    if (null == referenceLayer) return null;
    return new EquivalencyTester(1e-2, referenceLayer);
  }
  
  /**
   * Get input dims int [ ] [ ].
   *
   * @param random the random
   * @return the int [ ] [ ]
   */
  public abstract int[][] getInputDims(Random random);
  
  /**
   * Gets json tester.
   *
   * @return the json tester
   */
  protected ComponentTest<ToleranceStatistics> getJsonTester() {
    return new SerializationTest();
  }
  
  /**
   * Gets layer.
   *
   * @param inputSize the input size
   * @param random    the random
   * @return the layer
   */
  public abstract NNLayer getLayer(int[][] inputSize, Random random);
  
  /**
   * Gets little tests.
   *
   * @return the little tests
   */
  public ArrayList<ComponentTest<?>> getLittleTests() {
    if (null == littleTests) {
      synchronized (this) {
        if (null == littleTests) {
          littleTests = new ArrayList<>(Arrays.asList(
            getJsonTester(),
            getDerivativeTester()
                                                     ));
        }
      }
    }
    return littleTests;
  }
  
  /**
   * Get perf dims int [ ] [ ].
   *
   * @param random the random
   * @return the int [ ] [ ]
   */
  public int[][] getPerfDims(Random random) {
    return getInputDims(new Random());
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
   * Gets performance tester.
   *
   * @return the performance tester
   */
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    return new PerformanceTester();
  }
  
  /**
   * Gets reference io tester.
   *
   * @return the reference io tester
   */
  protected ComponentTest<ToleranceStatistics> getReferenceIOTester() {
    return new ReferenceIO(getReferenceIO());
  }
  
  /**
   * Gets reference layer.
   *
   * @return the reference layer
   */
  public NNLayer getReferenceLayer() {
    return cvt(getLayer(getInputDims(new Random()), new Random()));
  }
  
  /**
   * Gets test class.
   *
   * @return the test class
   */
  public Class<? extends NNLayer> getTestClass() {
    return getLayer(getInputDims(new Random()), new Random()).getClass();
  }
  
  /**
   * Cvt nn layer.
   *
   * @param layer the layer
   * @return the nn layer
   */
  protected NNLayer cvt(NNLayer layer) {
    if (layer instanceof DAGNetwork) {
      ((DAGNetwork) layer).visitNodes(node -> {
        NNLayer from = node.getLayer();
        node.setLayer(cvt(from));
      });
      return layer;
    }
    else if (getTestClass().isAssignableFrom(layer.getClass())) {
      Class<? extends NNLayer> referenceLayerClass = getReferenceLayerClass();
      if (null == referenceLayerClass) return null;
      return layer.as(referenceLayerClass);
    }
    else {
      return layer;
    }
  }
  
  /**
   * Gets reference layer class.
   *
   * @return the reference layer class
   */
  public Class<? extends NNLayer> getReferenceLayerClass() {
    return null;
  }
  
  /**
   * Gets learning tester.
   *
   * @return the learning tester
   */
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return isTestTraining() ? new TrainingTester() : null;
  }
  
  private final Random random = getRandom();
  
  /**
   * Random double.
   *
   * @return the double
   */
  public double random() {
    return random(random);
  }
  
  /**
   * Random double.
   *
   * @param random the random
   * @return the double
   */
  public double random(Random random) {
    return Math.round(1000.0 * (random.nextDouble() - 0.5)) / 250.0;
  }
  
  /**
   * Random tensor [ ].
   *
   * @param inputDims the input dims
   * @return the tensor [ ]
   */
  public Tensor[] randomize(final int[][] inputDims) {
    return Arrays.stream(inputDims).map(dim -> new Tensor(dim).set(() -> random())).toArray(i -> new Tensor[i]);
  }
  
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(final NotebookOutput log) {
    long seed = (long) (Math.random() * Long.MAX_VALUE);
    final NNLayer layer = getLayer(getInputDims(new Random(seed)), new Random(seed));
    if (layer instanceof DAGNetwork) {
      try {
        log.h1("Network Diagram");
        log.p("This is a network with the following layout:");
        log.code(() -> {
          return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
                         .height(400).width(600).render(Format.PNG).toImage();
        });
      } catch (Throwable e) {
        logger.info("Error plotting graph", e);
      }
    }
    else if (layer instanceof Explodable) {
      try {
        NNLayer explode = ((Explodable) layer).explode();
        if (explode instanceof DAGNetwork) {
          log.h1("Exploded Network Diagram");
          log.p("This is a network with the following layout:");
          log.code(() -> {
            return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) explode))
                           .height(400).width(600).render(Format.PNG).toImage();
          });
        }
      } catch (Throwable e) {
        logger.info("Error plotting graph", e);
      }
    }
    throwException(standardTests(log, seed));
    getFinalTests().stream().filter(x -> null != x).forEach(test -> {
      final NNLayer perfLayer = getLayer(getPerfDims(new Random(seed)), new Random(seed));
      test.test(log, perfLayer.copy(), randomize(getPerfDims(new Random(seed))));
    });
  }
  
  /**
   * Monte carlo.
   *
   * @param log the log
   */
  public void monteCarlo(final NotebookOutput log) {
    long timeout = System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(3);
    while (System.currentTimeMillis() < timeout) {
      long seed = (long) (Math.random() * Long.MAX_VALUE);
      final NNLayer layer = getLayer(getInputDims(new Random(seed)), new Random(seed));
      throwException(standardTests(log, seed));
    }
  }
  
  /**
   * Throw exception.
   *
   * @param exceptions the exceptions
   */
  public void throwException(ArrayList<Throwable> exceptions) {
    for (Throwable exception : exceptions) {
      throw new RuntimeException(exception);
    }
  }
  
  /**
   * Standard tests array list.
   *
   * @param log  the log
   * @param seed the seed
   * @return the array list
   */
  public ArrayList<Throwable> standardTests(NotebookOutput log, long seed) {
    final NNLayer layer = getLayer(getInputDims(new Random(seed)), new Random(seed));
    ArrayList<Throwable> exceptions = new ArrayList<>();
    log.p(String.format("Using Seed %d", seed));
    getLittleTests().stream().filter(x -> null != x).forEach((ComponentTest<?> test) -> {
      try {
        test.test(log, layer.copy(), randomize(getInputDims(new Random(seed))));
      } catch (Throwable e) { exceptions.add(e); }
    });
    getBigTests().stream().filter(x -> null != x).forEach(test -> {
      try {
        final NNLayer perfLayer = getLayer(getPerfDims(new Random(seed)), new Random(seed));
        test.test(log, perfLayer.copy(), randomize(getPerfDims(new Random(seed))));
      } catch (Throwable e) { exceptions.add(e); }
    });
    return exceptions;
  }
  
  /**
   * With big tests standard layer tests.
   *
   * @param fn the fn
   * @return the standard layer tests
   */
  public StandardLayerTests withBigTests(final Consumer<ArrayList<ComponentTest<?>>> fn) {
    fn.accept(getBigTests());
    return this;
  }
  
  /**
   * With little tests standard layer tests.
   *
   * @param fn the fn
   * @return the standard layer tests
   */
  public StandardLayerTests withLittleTests(final Consumer<ArrayList<ComponentTest<?>>> fn) {
    fn.accept(getLittleTests());
    return this;
  }
  
  @Override
  protected Class<?> getTargetClass() {
    return getLayer(getInputDims(new Random()), new Random()).getClass();
  }
  
  @Override
  public ReportType getReportType() {
    return ReportType.Components;
  }
  
  /**
   * Is test training boolean.
   *
   * @return the boolean
   */
  public boolean isTestTraining() {
    return testTraining;
  }
  
  /**
   * Sets test training.
   *
   * @param testTraining the test training
   * @return the test training
   */
  public StandardLayerTests setTestTraining(boolean testTraining) {
    this.testTraining = testTraining;
    return this;
  }
  
  /**
   * Gets random.
   *
   * @return the random
   */
  public Random getRandom() {
    return new Random(seed);
  }
}
