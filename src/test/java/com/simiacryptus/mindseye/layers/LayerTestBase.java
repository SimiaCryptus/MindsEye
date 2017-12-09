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

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.java.ActivationLayerTestBase;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import guru.nidi.graphviz.attribute.RankDir;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.*;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.PrintStream;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Layer test base.
 */
public abstract class LayerTestBase {
  /**
   * The constant originalOut.
   */
  protected static final PrintStream originalOut = System.out;
  private static final Logger log = LoggerFactory.getLogger(ActivationLayerTestBase.class);
  
  /**
   * To graph graph.
   *
   * @param network the network
   * @return the graph
   */
  public static Graph toGraph(DAGNetwork network) {
    List<DAGNode> nodes = network.getNodes();
    Map<UUID, MutableNode> graphNodes = nodes.stream().collect(Collectors.toMap(node -> node.getId(), node -> {
      String name;
      NNLayer layer = node.getLayer();
      if (null == layer) {
        name = node.getId().toString();
      }
      else {
        Class<? extends NNLayer> layerClass = layer.getClass();
        name = layerClass.getSimpleName() + "\n" + layer.getId();
      }
      return Factory.mutNode(name);
    }));
    Stream<UUID[]> stream = nodes.stream().flatMap(to -> {
      return Arrays.stream(to.getInputs()).map(from -> {
        return new UUID[]{from.getId(), to.getId()};
      });
    });
    Map<UUID, List<UUID>> idMap = stream.collect(Collectors.groupingBy(x -> x[0],
      Collectors.mapping(x -> x[1], Collectors.toList())));
    nodes.forEach(to -> {
      graphNodes.get(to.getId()).addLink(
        idMap.getOrDefault(to.getId(), Arrays.asList()).stream().map(from -> {
          return Link.to(graphNodes.get(from));
        }).<LinkTarget>toArray(i -> new LinkTarget[i]));
    });
    LinkSource[] nodeArray = graphNodes.values().stream().map(x -> (LinkSource) x).toArray(i -> new LinkSource[i]);
    return Factory.graph().with(nodeArray).generalAttr().with(RankDir.TOP_TO_BOTTOM).directed();
  }
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void test() throws Throwable {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      test(log);
    }
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void test(NotebookOutput log) {
    if (null != originalOut) log.addCopy(originalOut);
    NNLayer layer = getLayer();
    log.h1("%s", layer.getClass().getSimpleName());
    log.h2("%s", getClass().getSimpleName());
    log.h3("Json Serialization");
    log.code(() -> {
      JsonObject json = layer.getJson();
      NNLayer echo = NNLayer.fromJson(json);
      assert (echo != null) : "Failed to deserialize";
      assert (layer != echo) : "Serialization did not copy";
      Assert.assertEquals("Serialization not equal", layer, echo);
      return new GsonBuilder().setPrettyPrinting().create().toJson(json);
    });
    
    if (layer instanceof DAGNetwork) {
      log.h3("Network Diagram");
      log.code(() -> {
        return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
          .height(400).width(600).render(Format.PNG).toImage();
      });
    }
    
    Tensor[] inputPrototype = Arrays.stream(getInputDims()).map(dim -> new Tensor(dim).fill(() -> random()))
      .toArray(i -> new Tensor[i]);
    Tensor outputPrototype = GpuController.INSTANCE.distribute(Arrays.<Tensor[]>asList(inputPrototype),
      (data, exe) -> layer.eval(exe, NNResult.batchResultArray(data.toArray(new Tensor[][]{}))).getData().get(0),
      (a, b) -> a.add(b));
    
    HashMap<Tensor[], Tensor> referenceIO = getReferenceIO();
    if (!referenceIO.isEmpty()) {
      log.h3("Reference Input/Output Pairs");
      referenceIO.forEach((input, output) -> {
        log.code(() -> {
          SimpleEval eval = SimpleEval.run(layer, input);
          DoubleStatistics error = new DoubleStatistics().accept(eval.getOutput().add(output.scale(-1)).getData());
          return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\nError: %s",
            Arrays.stream(input).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
            eval.getOutput().prettyPrint(), error);
        });
      });
    }
    else {
      log.h3("Example Input/Output Pair");
      log.code(() -> {
        SimpleEval eval = SimpleEval.run(layer, inputPrototype);
        return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
          Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
          eval.getOutput().prettyPrint());
      });
    }
    
    NNLayer referenceLayer = getReferenceLayer();
    if (null != referenceLayer) {
      log.h3("Reference Implementation");
      log.code(() -> {
        System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
        getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
      });
    }
    
    log.h3("Batch Execution");
    log.code(() -> {
      BatchingTester batchingTester = getBatchingTester();
      return batchingTester == null ? null : batchingTester.test(layer, inputPrototype);
    });
    
    log.h3("Differential Validation");
    log.code(() -> {
      return getDerivativeTester().test(layer, inputPrototype);
    });
    
    log.h3("Performance");
    log.code(() -> {
      getPerformanceTester().test(layer, inputPrototype);
    });
    
  }
  
  /**
   * Gets batching tester.
   *
   * @return the batching tester
   */
  public BatchingTester getBatchingTester() {
    return new BatchingTester(1e-2);
  }
  
  /**
   * Random double.
   *
   * @return the double
   */
  public double random() {
    return Math.round(1000 * (Util.R.get().nextDouble() - 0.5)) / 250.0;
  }
  
  /**
   * Gets equivalency tester.
   *
   * @return the equivalency tester
   */
  public EquivalencyTester getEquivalencyTester() {
    return new EquivalencyTester(1e-2);
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
  public DerivativeTester getDerivativeTester() {
    return new DerivativeTester(1e-3, 1e-4);
  }
  
  /**
   * Gets layer.
   *
   * @return the layer
   */
  public abstract NNLayer getLayer();
  
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
}
