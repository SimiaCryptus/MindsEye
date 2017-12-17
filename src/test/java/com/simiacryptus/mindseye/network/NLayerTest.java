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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.unit.JsonTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;


/**
 * The type N layer test.
 */
public abstract class NLayerTest {
  
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
    if (null != LayerTestBase.originalOut) log.addCopy(LayerTestBase.originalOut);
    log.h1("%s", getClass().getSimpleName());
    int[] inputDims = getInputDims();
    ArrayList<int[]> workingSpec = new ArrayList<>();
    for(int[] l : this.dimList) {
      workingSpec.add(l);
      NNLayer layer = buildNetwork(concat(inputDims, workingSpec));
      graphviz(log, layer);
      test(log, layer, inputDims);
    }
  }
  
  /**
   * Test double.
   *
   * @param log       the log
   * @param layer     the layer
   * @param inputDims the input dims
   * @return the double
   */
  public TrainingTester.ComponentResult test(NotebookOutput log, NNLayer layer, int[]... inputDims) {
    NNLayer component = layer.copy();
    Tensor[] randomize = randomize(inputDims);
    new JsonTest().test(log, component, randomize);
    return new TrainingTester().test(log, component, randomize);
  }
  
  /**
   * Graphviz.
   *
   * @param log   the log
   * @param layer the layer
   */
  public void graphviz(NotebookOutput log, NNLayer layer) {
    if (layer instanceof DAGNetwork) {
      log.p("This is a network with the following layout:");
      log.code(() -> {
        return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
          .height(400).width(600).render(Format.PNG).toImage();
      });
    }
  }
  
  /**
   * Concat int [ ] [ ].
   *
   * @param a the a
   * @param b the b
   * @return the int [ ] [ ]
   */
  public int[][] concat(int[] a, List<int[]> b) {
    return Stream.concat(Stream.of(a), b.stream()).toArray(i -> new int[i][]);
  }
  
  /**
   * Random tensor [ ].
   *
   * @param inputDims the input dims
   * @return the tensor [ ]
   */
  public Tensor[] randomize(int[][] inputDims) {
    return Arrays.stream(inputDims).map(dim -> new Tensor(dim).fill(this::random)).toArray(i -> new Tensor[i]);
  }
  
  /**
   * Get input dims int [ ] [ ].
   *
   * @return the int [ ] [ ]
   */
  public abstract int[] getInputDims();
  
  /**
   * Random double.
   *
   * @return the double
   */
  public double random() {
    return Math.round(1000.0 * (Util.R.get().nextDouble() - 0.5)) / 250.0;
  }
  
  
  /**
   * The Dim list.
   */
  final List<int[]> dimList;
  
  /**
   * Instantiates a new N layer test.
   *
   * @param dimList the dim list
   */
  public NLayerTest(int[]... dimList) {
    this.dimList = Arrays.asList(dimList);
  }
  
  /**
   * Build network nn layer.
   *
   * @param dimList the dim list
   * @return the nn layer
   */
  public NNLayer buildNetwork(int[]... dimList) {
    PipelineNetwork network = new PipelineNetwork(1);
    int[] last = null;
    for (int[] dims : dimList) {
      if(null != last) addLayer(network, last, dims);
      last = dims;
    }
    return network;
  }
  
  /**
   * Add layer.
   *
   * @param network the network
   * @param in      the in
   * @param out     the dims
   */
  public abstract void addLayer(PipelineNetwork network, int[] in, int[] out);
  
}
