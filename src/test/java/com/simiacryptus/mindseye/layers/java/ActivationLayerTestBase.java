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

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.unit.ComponentTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;
import com.simiacryptus.util.io.NotebookOutput;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * The type Activation layer run base.
 */
public abstract class ActivationLayerTestBase extends LayerTestBase {
  
  private final NNLayer layer;
  
  /**
   * Instantiates a new Activation layer run base.
   *
   * @param layer the layer
   */
  public ActivationLayerTestBase(final NNLayer layer) {
    this.layer = layer;
  }
  
  /**
   * Plot plot canvas.
   *
   * @param title the title
   * @param data  the data
   * @return the plot canvas
   */
  public static @NotNull PlotCanvas plot(final String title, final double[][] data) {
    final @NotNull PlotCanvas plot = ScatterPlot.plot(data);
    plot.setTitle(title);
    plot.setAxisLabels("x", "y");
    plot.setSize(600, 400);
    return plot;
  }
  
  /**
   * Plot plot canvas.
   *
   * @param title    the title
   * @param plotData the plot data
   * @param function the function
   * @return the plot canvas
   */
  public static @NotNull PlotCanvas plot(final String title, final @NotNull List<double[]> plotData, final Function<double[], double[]> function) {
    final double[][] data = plotData.stream().map(function).toArray(i -> new double[i][]);
    return ActivationLayerTestBase.plot(title, data);
  }
  
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {2, 3, 1}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return layer;
  }
  
  @Override
  public @NotNull int[][] getLargeDims(Random random) {
    return new int[][]{
      {100, 100, 1}
    };
  }
  
  /**
   * Scan double stream.
   *
   * @return the double stream
   */
  public DoubleStream scan() {
    return IntStream.range(-1000, 1000).mapToDouble(x -> x / 300.0);
  }
  
  @Override
  public void run(final NotebookOutput log) {
    super.run(log);
    
    log.h3("Function Plots");
    final NNLayer layer = getLayer(new int[][]{{1}}, new Random());
    final List<double[]> plotData = scan().mapToObj(x -> {
      final @NotNull SimpleEval eval = SimpleEval.run(layer, new Tensor(x));
      @NotNull double[] doubles = {x, eval.getOutput().get(0), eval.getDerivative()[0].get(0)};
      eval.freeRef();
      return doubles;
    }).collect(Collectors.toList());
    
    log.code(() -> {
      return ActivationLayerTestBase.plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
    });
    
    log.code(() -> {
      return ActivationLayerTestBase.plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
    });
    
  }
  
  @Override
  public @Nullable ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
    return new TrainingTester().setRandomizationMode(TrainingTester.RandomizationMode.Random);
  }
}
