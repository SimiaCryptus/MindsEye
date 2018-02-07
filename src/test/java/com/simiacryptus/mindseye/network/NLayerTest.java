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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.unit.SerializationTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.SysOutInterceptor;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;


/**
 * The type N layer eval.
 */
public abstract class NLayerTest {
  static {
    SysOutInterceptor.INSTANCE.init();
  }
  
  
  /**
   * The Dim list.
   */
  final @NotNull List<int[]> dimList;
  
  
  /**
   * Instantiates a new N layer eval.
   *
   * @param dimList the dim list
   */
  public NLayerTest(final int[]... dimList) {
    this.dimList = Arrays.asList(dimList);
  }
  
  /**
   * Add layer.
   *
   * @param network the network
   * @param in      the in
   * @param out     the dims
   */
  public abstract void addLayer(PipelineNetwork network, int[] in, int[] out);
  
  /**
   * Build network nn layer.
   *
   * @param dimList the dim list
   * @return the nn layer
   */
  public @NotNull NNLayer buildNetwork(final @NotNull int[]... dimList) {
    final @NotNull PipelineNetwork network = new PipelineNetwork(1);
    @Nullable int[] last = null;
    for (final int[] dims : dimList) {
      if (null != last) {
        addLayer(network, last, dims);
      }
      last = dims;
    }
    return network;
  }
  
  /**
   * Concat int [ ] [ ].
   *
   * @param a the a
   * @param b the b
   * @return the int [ ] [ ]
   */
  public int[][] concat(final int[] a, final @NotNull List<int[]> b) {
    return Stream.concat(Stream.of(a), b.stream()).toArray(i -> new int[i][]);
  }
  
  /**
   * Get input dims int [ ] [ ].
   *
   * @return the int [ ] [ ]
   */
  public abstract int[] getInputDims();
  
  /**
   * Graphviz.
   *
   * @param log   the log
   * @param layer the layer
   */
  public void graphviz(final @NotNull NotebookOutput log, final NNLayer layer) {
    if (layer instanceof DAGNetwork) {
      log.p("This is a network with the following layout:");
      log.code(() -> {
        return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
                       .height(400).width(600).render(Format.PNG).toImage();
      });
    }
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
   * Random tensor [ ].
   *
   * @param inputDims the input dims
   * @return the tensor [ ]
   */
  public Tensor[] randomize(final @NotNull int[][] inputDims) {
    return Arrays.stream(inputDims).map(dim -> new Tensor(dim).set(this::random)).toArray(i -> new Tensor[i]);
  }
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void test() throws Throwable {
    try (@NotNull NotebookOutput log = MarkdownNotebookOutput.get(((Object) this).getClass(), null)) {
      test(log);
    }
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void test(final @NotNull NotebookOutput log) {
  
    log.h1("%s", getClass().getSimpleName());
    final int[] inputDims = getInputDims();
    final @NotNull ArrayList<int[]> workingSpec = new ArrayList<>();
    for (final int[] l : dimList) {
      workingSpec.add(l);
      final @NotNull NNLayer layer = buildNetwork(concat(inputDims, workingSpec));
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
  public @Nullable TrainingTester.ComponentResult test(final @NotNull NotebookOutput log, final @NotNull NNLayer layer, final @NotNull int[]... inputDims) {
    final NNLayer component = layer.copy();
    final Tensor[] randomize = randomize(inputDims);
    new SerializationTest().test(log, component, randomize);
    return new TrainingTester().test(log, component, randomize);
  }
  
}
