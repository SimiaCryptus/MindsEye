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
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * The type N layer eval.
 */
public abstract class PipelineTest {
  
  /**
   * The Dim list.
   */
  final List<NNLayer> pipeline;
  
  
  /**
   * Instantiates a new N layer eval.
   *
   * @param pipeline the pipeline
   */
  public PipelineTest(final List<NNLayer> pipeline) {
    this.pipeline = pipeline;
  }
  
  /**
   * Instantiates a new Pipeline eval.
   *
   * @param pipeline the pipeline
   */
  public PipelineTest(final NNLayer... pipeline) {
    this(Arrays.asList(pipeline));
  }
  
  /**
   * Build network nn layer.
   *
   * @param layers the dim list
   * @return the nn layer
   */
  @javax.annotation.Nonnull
  public NNLayer buildNetwork(@javax.annotation.Nonnull final NNLayer... layers) {
    @javax.annotation.Nonnull final PipelineNetwork network = new PipelineNetwork(1);
    for (@javax.annotation.Nonnull final NNLayer layer : layers) {
      network.add(layer.copy());
    }
    return network;
  }
  
  /**
   * Get input dims int [ ] [ ].
   *
   * @return the int [ ] [ ]
   */
  @Nonnull
  public abstract int[] getInputDims();
  
  /**
   * Graphviz.
   *
   * @param log   the log
   * @param layer the layer
   */
  public void graphviz(@javax.annotation.Nonnull final NotebookOutput log, final NNLayer layer) {
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
  public Tensor[] randomize(@javax.annotation.Nonnull final int[][] inputDims) {
    return Arrays.stream(inputDims).map(dim -> new Tensor(dim).set(this::random)).toArray(i -> new Tensor[i]);
  }
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void test() throws Throwable {
    try (@javax.annotation.Nonnull NotebookOutput log = MarkdownNotebookOutput.get(((Object) this).getClass(), null)) {
      test(log);
    }
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void test(@javax.annotation.Nonnull final NotebookOutput log) {
    @javax.annotation.Nonnull final ArrayList<NNLayer> workingSpec = new ArrayList<>();
    int layerIndex = 0;
    for (final NNLayer l : pipeline) {
      workingSpec.add(l);
      @javax.annotation.Nonnull final NNLayer networkHead = buildNetwork(workingSpec.toArray(new NNLayer[]{}));
      graphviz(log, networkHead);
      test(log, networkHead, String.format("Pipeline Network with %d Layers", layerIndex++), getInputDims());
    }
  }
  
  /**
   * Test double.
   *
   * @param log       the log
   * @param layer     the layer
   * @param header    the header
   * @param inputDims the input dims
   * @return the double
   */
  @Nullable
  public TrainingTester.ComponentResult test(@javax.annotation.Nonnull final NotebookOutput log, @javax.annotation.Nonnull final NNLayer layer, final String header, @javax.annotation.Nonnull final int[]... inputDims) {
    @Nonnull final NNLayer component = layer.copy();
    final Tensor[] randomize = randomize(inputDims);
    new SerializationTest().test(log, component, randomize);
    return new TrainingTester() {
      @Override
      protected void printHeader(NotebookOutput log) {
        log.h1(header);
      }
    }.test(log, component, randomize);
  }
  
}
