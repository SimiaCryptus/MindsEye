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

import com.google.gson.JsonObject;
import com.simiacryptus.devutil.Javadoc;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.CudaError;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.util.IOUtil;
import com.simiacryptus.util.test.SysOutInterceptor;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * The type LayerBase apply base.
 */
public abstract class StandardLayerTests extends NotebookReportBase {
  /**
   * The constant seed.
   */
  public static final long seed = 51389; //System.nanoTime();
  private static final HashMap<String, TreeMap<String, String>> javadocs = loadJavadoc();

  static {
    SysOutInterceptor.INSTANCE.init();
  }

  private final Random random = getRandom();
  /**
   * The Testing batch size.
   */
  protected int testingBatchSize = 5;
  /**
   * The Validate batch execution.
   */
  protected boolean validateBatchExecution = true;
  /**
   * The Validate differentials.
   */
  protected boolean validateDifferentials = true;
  /**
   * The Test training.
   */
  protected boolean testTraining = true;
  /**
   * The Test equivalency.
   */
  protected boolean testEquivalency = true;
  /**
   * The Tolerance.
   */
  protected double tolerance;

  /**
   * Instantiates a new Standard key tests.
   */
  public StandardLayerTests() {
    logger.info("Seed: " + seed);
    tolerance = 1e-3;
  }

  @Nonnull
  private static HashMap<String, TreeMap<String, String>> loadJavadoc() {
    try {
      HashMap<String, TreeMap<String, String>> javadocData = Javadoc.loadModelSummary();
      IOUtil.writeJson(new TreeMap<>(javadocData), new File("./javadoc.json"));
      return javadocData;
    } catch (Throwable e) {
      logger.warn("Error loading javadocs", e);
      return new HashMap<>();
    }
  }

  /**
   * Gets batching tester.
   *
   * @return the batching tester
   */
  @Nullable
  public ComponentTest<ToleranceStatistics> getBatchingTester() {
    if (!validateBatchExecution) return null;
    return new BatchingTester(1e-2) {
      @Override
      public double getRandom() {
        return random();
      }
    }.setBatchSize(testingBatchSize);
  }

  /**
   * Gets big tests.
   *
   * @return the big tests
   */
  @Nonnull
  public List<ComponentTest<?>> getBigTests() {
    return Arrays.asList(
        getPerformanceTester(),
        getBatchingTester(),
        getReferenceIOTester(),
        getEquivalencyTester()
    );
  }

  /**
   * Gets big tests.
   *
   * @return the big tests
   */
  @Nonnull
  public List<ComponentTest<?>> getFinalTests() {
    return Arrays.asList(
        getTrainingTester()
    );
  }

  /**
   * Gets derivative tester.
   *
   * @return the derivative tester
   */
  @Nullable
  public ComponentTest<ToleranceStatistics> getDerivativeTester() {
    if (!validateDifferentials) return null;
    return new SingleDerivativeTester(tolerance, 1e-4);
  }

  /**
   * Gets equivalency tester.
   *
   * @return the equivalency tester
   */
  @Nullable
  public ComponentTest<ToleranceStatistics> getEquivalencyTester() {
    if (!testEquivalency) return null;
    @Nullable final Layer referenceLayer = getReferenceLayer();
    if (null == referenceLayer) return null;
    @Nonnull EquivalencyTester equivalencyTester = new EquivalencyTester(1e-2, referenceLayer);
    referenceLayer.freeRef();
    return equivalencyTester;
  }

  /**
   * Get input dims int [ ] [ ].
   *
   * @param random the random
   * @return the int [ ] [ ]
   */
  public abstract int[][] getSmallDims(Random random);

  /**
   * Gets json tester.
   *
   * @return the json tester
   */
  @Nullable
  protected ComponentTest<ToleranceStatistics> getJsonTester() {
    return new SerializationTest();
  }

  /**
   * Gets key.
   *
   * @param inputSize the input size
   * @param random    the random
   * @return the key
   */
  public abstract Layer getLayer(int[][] inputSize, Random random);

  /**
   * Gets little tests.
   *
   * @return the little tests
   */
  @Nonnull
  public List<ComponentTest<?>> getLittleTests() {
    return Arrays.asList(
        getJsonTester(),
        getDerivativeTester()
    );
  }

  /**
   * Get perf dims int [ ] [ ].
   *
   * @param random the random
   * @return the int [ ] [ ]
   */
  public int[][] getLargeDims(Random random) {
    return getSmallDims(new Random());
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
  @Nullable
  public ComponentTest<ToleranceStatistics> getPerformanceTester() {
    return new PerformanceTester();
  }

  /**
   * Gets reference io tester.
   *
   * @return the reference io tester
   */
  @Nullable
  protected ComponentTest<ToleranceStatistics> getReferenceIOTester() {
    return new ReferenceIO(getReferenceIO());
  }

  /**
   * Gets reference key.
   *
   * @return the reference key
   */
  @Nullable
  public Layer getReferenceLayer() {
    return cvt(getLayer(getSmallDims(new Random()), new Random()));
  }

  /**
   * Gets test class.
   *
   * @return the test class
   */
  public Class<? extends Layer> getTestClass() {
    Layer layer = getLayer(getSmallDims(new Random()), new Random());
    Class<? extends Layer> layerClass = layer.getClass();
    layer.freeRef();
    return layerClass;
  }

  /**
   * Cvt nn key.
   *
   * @param layer the key
   * @return the nn key
   */
  protected final Layer cvt(Layer layer) {
    if (layer instanceof DAGNetwork) {
      ((DAGNetwork) layer).visitNodes(node -> {
        @Nullable Layer from = node.getLayer();
        node.setLayer(cvt(from));
      });
      return layer;
    } else if (getTestClass().isAssignableFrom(layer.getClass())) {
      @Nullable Class<? extends Layer> referenceLayerClass = getReferenceLayerClass();
      if (null == referenceLayerClass) {
        layer.freeRef();
        return null;
      } else {
        @Nonnull Layer cast = layer.as(referenceLayerClass);
        layer.freeRef();
        return cast;
      }
    } else {
      return layer;
    }
  }

  /**
   * Gets reference key class.
   *
   * @return the reference key class
   */
  @Nullable
  public Class<? extends Layer> getReferenceLayerClass() {
    return null;
  }

  /**
   * Gets learning tester.
   *
   * @return the learning tester
   */
  @Nullable
  public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return isTestTraining() ? new TrainingTester() : null;
  }

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
  public double random(@Nonnull Random random) {
    return Math.round(1000.0 * (random.nextDouble() - 0.5)) / 250.0;
  }

  /**
   * Random tensor [ ].
   *
   * @param inputDims the input dims
   * @return the tensor [ ]
   */
  public Tensor[] randomize(@Nonnull final int[][] inputDims) {
    return Arrays.stream(inputDims).map(dim -> new Tensor(dim).set(() -> random())).toArray(i -> new Tensor[i]);
  }

  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull final NotebookOutput log) {
    TreeMap<String, String> javadoc = javadocs.get(getTargetClass().getCanonicalName());
    if (null != javadoc) {
      log.p("Class Javadoc: " + javadoc.get(":class"));
      javadoc.remove(":class");
      javadoc.forEach((key, doc) -> {
        log.p(String.format("Field __%s__: %s", key, doc));
      });
    }

    long seed = (long) (Math.random() * Long.MAX_VALUE);
    int[][] smallDims = getSmallDims(new Random(seed));
    final Layer smallLayer = getLayer(smallDims, new Random(seed));
    int[][] largeDims = getLargeDims(new Random(seed));
    final Layer largeLayer = getLayer(largeDims, new Random(seed));

    log.h1("Test Modules");
    TableOutput results = new TableOutput();
    try {
      if (smallLayer instanceof DAGNetwork) {
        try {
          log.h1("Network Diagram");
          log.p("This is a network apply the following layout:");
          log.eval(() -> {
            return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) smallLayer))
                .height(400).width(600).render(Format.PNG).toImage();
          });
        } catch (Throwable e) {
          logger.info("Error plotting graph", e);
        }
      } else if (smallLayer instanceof Explodable) {
        try {
          Layer explode = ((Explodable) smallLayer).explode();
          if (explode instanceof DAGNetwork) {
            log.h1("Exploded Network Diagram");
            log.p("This is a network apply the following layout:");
            @Nonnull DAGNetwork network = (DAGNetwork) explode;
            log.eval(() -> {
              @Nonnull Graphviz graphviz = Graphviz.fromGraph(TestUtil.toGraph(network)).height(400).width(600);
              @Nonnull File file = new File(log.getResourceDir(), log.getName() + "_network.svg");
              graphviz.render(Format.SVG_STANDALONE).toFile(file);
              log.link(file, "Saved to File");
              return graphviz.render(Format.SVG).toString();
            });
          }
        } catch (Throwable e) {
          logger.info("Error plotting graph", e);
        }
      }
      @Nonnull ArrayList<TestError> exceptions = standardTests(log, seed, results);
      if (!exceptions.isEmpty()) {
        if (smallLayer instanceof DAGNetwork) {
          for (@Nonnull Invocation invocation : getInvocations(smallLayer, smallDims)) {
            log.h1("Small SubTests: " + invocation.getLayer().getClass().getSimpleName());
            log.p(Arrays.deepToString(invocation.getDims()));
            tests(log, getLittleTests(), invocation, exceptions, results);
            invocation.freeRef();
          }
        }
        if (largeLayer instanceof DAGNetwork) {
          testEquivalency = false;
          for (@Nonnull Invocation invocation : getInvocations(largeLayer, largeDims)) {
            log.h1("Large SubTests: " + invocation.getLayer().getClass().getSimpleName());
            log.p(Arrays.deepToString(invocation.getDims()));
            tests(log, getBigTests(), invocation, exceptions, results);
            invocation.freeRef();
          }
        }
      }
      log.run(() -> {
        throwException(exceptions);
      });
    } finally {
      smallLayer.freeRef();
      largeLayer.freeRef();
    }
    getFinalTests().stream().filter(x -> null != x).forEach(test -> {
      final Layer perfLayer;
      perfLayer = getLayer(largeDims, new Random(seed));
      perfLayer.assertAlive();
      @Nonnull Layer copy;
      copy = perfLayer.copy();
      Tensor[] randomize = randomize(largeDims);
      HashMap<CharSequence, Object> testResultProps = new HashMap<>();
      try {
        String testclass = test.getClass().getCanonicalName();
        testResultProps.put("class", testclass);
        Object result = log.subreport(testclass, sublog->test.test(sublog, copy, randomize));
        testResultProps.put("details", null==result?null:result.toString());
        testResultProps.put("result", "OK");
      } catch (Throwable e) {
        testResultProps.put("result", e.toString());
        throw new RuntimeException(e);
      } finally {
        results.putRow(testResultProps);
        test.freeRef();
        for (@Nonnull Tensor tensor : randomize) {
          tensor.freeRef();
        }
        perfLayer.freeRef();
        copy.freeRef();
      }
    });
    log.h1("Test Matrix");
    log.out(results.toMarkdownTable());
  }

  /**
   * Gets invocations.
   *
   * @param smallLayer the small key
   * @param smallDims  the small dims
   * @return the invocations
   */
  @Nonnull
  public Collection<Invocation> getInvocations(@Nonnull Layer smallLayer, @Nonnull int[][] smallDims) {
    @Nonnull DAGNetwork smallCopy = (DAGNetwork) smallLayer.copy();
    @Nonnull HashSet<Invocation> invocations = new HashSet<>();
    smallCopy.visitNodes(node -> {
      @Nullable Layer inner = node.getLayer();
      inner.addRef();
      @Nullable Layer wrapper = new LayerBase() {
        @Nullable
        @Override
        public Result eval(@Nonnull Result... array) {
          if (null == inner) return null;
          @Nullable Result result = inner.eval(array);
          invocations.add(new Invocation(inner, Arrays.stream(array).map(x -> x.getData().getDimensions()).toArray(i -> new int[i][])));
          return result;
        }

        @Override
        public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
          return inner.getJson(resources, dataSerializer);
        }

        @Nullable
        @Override
        public List<double[]> state() {
          return inner.state();
        }

        @Override
        protected void _free() {
          inner.freeRef();
        }
      };
      node.setLayer(wrapper);
      wrapper.freeRef();
    });
    Tensor[] input = Arrays.stream(smallDims).map(i -> new Tensor(i)).toArray(i -> new Tensor[i]);
    try {
      Result eval = smallCopy.eval(input);
      eval.freeRef();
      eval.getData().freeRef();
      return invocations;
    } finally {
      Arrays.stream(input).forEach(ReferenceCounting::freeRef);
      smallCopy.freeRef();
    }
  }

  /**
   * Throw exception.
   *
   * @param exceptions the exceptions
   */
  public void throwException(@Nonnull ArrayList<TestError> exceptions) {
    for (@Nonnull TestError exception : exceptions) {
      logger.info(String.format("LayerBase: %s", exception.layer));
      logger.info("Error", exception);
    }
    for (Throwable exception : exceptions) {
      try {
        ReferenceCountingBase.supressLog = true;
        System.gc();
        throw new RuntimeException(exception);
      } finally {
        ReferenceCountingBase.supressLog = false;
      }
    }
  }

  /**
   * Standard tests array list.
   *
   * @param log  the log
   * @param seed the seed
   * @param results
   * @return the array list
   */
  @Nonnull
  public ArrayList<TestError> standardTests(@Nonnull NotebookOutput log, long seed, TableOutput results) {
    log.p(String.format("Using Seed %d", seed));
    @Nonnull ArrayList<TestError> exceptions = new ArrayList<>();
    final Layer layer = getLayer(getSmallDims(new Random(seed)), new Random(seed));
    Invocation invocation = new Invocation(layer, getSmallDims(new Random(seed)));
    try {
      tests(log, getLittleTests(), invocation, exceptions, results);
    } finally {
      invocation.freeRef();
      layer.freeRef();
    }
    final Layer perfLayer = getLayer(getLargeDims(new Random(seed)), new Random(seed));
    try {
      bigTests(log, seed, perfLayer, exceptions, results);
    } finally {
      perfLayer.freeRef();
    }
    return exceptions;
  }

  /**
   * Big tests.
   *  @param log        the log
   * @param seed       the seed
   * @param perfLayer  the perf key
   * @param exceptions the exceptions
   * @param results
   */
  public void bigTests(NotebookOutput log, long seed, @Nonnull Layer perfLayer, @Nonnull ArrayList<TestError> exceptions, TableOutput results) {
    getBigTests().stream().filter(x -> null != x).forEach(test -> {
      @Nonnull Layer layer = perfLayer.copy();
      try {
        Tensor[] input = randomize(getLargeDims(new Random(seed)));
        LinkedHashMap<CharSequence, Object> testResultProps = new LinkedHashMap<>();
        try {
          String testclass = test.getClass().getCanonicalName();
          if(null == testclass || testclass.isEmpty()) testclass = test.toString();
          testResultProps.put("class", testclass);
          Object result = log.subreport(testclass, sublog->test.test(sublog, layer, input));
          testResultProps.put("details", null==result?null:result.toString());
          testResultProps.put("result", "OK");
        } catch (Throwable e){
          testResultProps.put("result", e.toString());
          throw new RuntimeException(e);
        } finally {
          results.putRow(testResultProps);
          for (@Nonnull Tensor t : input) {
            t.freeRef();
          }
        }
      } catch (LifecycleException e) {
        throw e;
      } catch (CudaError e) {
        throw e;
      } catch (Throwable e) {
        exceptions.add(new TestError(e, test, layer));
      } finally {
        layer.freeRef();
        test.freeRef();
        System.gc();
      }
    });
  }

  private void tests(final NotebookOutput log, final List<ComponentTest<?>> tests, @Nonnull final Invocation invocation, @Nonnull final ArrayList<TestError> exceptions, TableOutput results) {
    tests.stream().filter(x -> null != x).forEach((ComponentTest<?> test) -> {
      @Nonnull Layer layer = invocation.getLayer().copy();
      layer.addRef();
      Tensor[] inputs = randomize(invocation.getDims());
      LinkedHashMap<CharSequence, Object> testResultProps = new LinkedHashMap<>();
      try {
        String testname = test.getClass().getCanonicalName();
        testResultProps.put("class", testname);
        Object result = log.subreport(testname, sublog->test.test(sublog, layer, inputs));
        testResultProps.put("details", null==result?null:result.toString());
        testResultProps.put("result", "OK");
      } catch (LifecycleException e) {
        throw e;
      } catch (Throwable e) {
        testResultProps.put("result", e.toString());
        exceptions.add(new TestError(e, test, layer));
      } finally {
        results.putRow(testResultProps);
        for (@Nonnull Tensor tensor : inputs) tensor.freeRef();
        layer.freeRef();
        test.freeRef();
        System.gc();
      }
    });
  }

  @Override
  protected Class<?> getTargetClass() {
    Layer layer = getLayer(getSmallDims(new Random()), new Random());
    Class<? extends Layer> layerClass = layer.getClass();
    layer.freeRef();
    return layerClass;
  }

  @Nonnull
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
  @Nonnull
  public StandardLayerTests setTestTraining(boolean testTraining) {
    this.testTraining = testTraining;
    return this;
  }

  /**
   * Gets random.
   *
   * @return the random
   */
  @Nonnull
  public Random getRandom() {
    return new Random(seed);
  }

  private static class Invocation extends ReferenceCountingBase {
    private final Layer layer;
    private final int[][] smallDims;

    private Invocation(Layer layer, int[][] smallDims) {
      this.layer = layer;
      this.smallDims = smallDims;
      this.layer.addRef();
    }

    @Override
    protected void _free() {
      this.layer.freeRef();
      super._free();
    }

    /**
     * Gets key.
     *
     * @return the key
     */
    public Layer getLayer() {
      return layer;
    }

    /**
     * Get dims int [ ] [ ].
     *
     * @return the int [ ] [ ]
     */
    public int[][] getDims() {
      return smallDims;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Invocation)) return false;

      @Nonnull Invocation that = (Invocation) o;

      if (layer != null ? !layer.getClass().equals(that.layer.getClass()) : that.layer != null) return false;
      return Arrays.deepEquals(smallDims, that.smallDims);
    }

    @Override
    public int hashCode() {
      int result = layer != null ? layer.getClass().hashCode() : 0;
      result = 31 * result + Arrays.deepHashCode(smallDims);
      return result;
    }
  }
}
