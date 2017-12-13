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

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.util.io.NotebookOutput;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * The type Activation layer test base.
 */
public abstract class ActivationLayerTestBase extends LayerTestBase {
  
  private final NNLayer layer;
  
  /**
   * Instantiates a new Activation layer test base.
   *
   * @param layer the layer
   */
  public ActivationLayerTestBase(NNLayer layer) {
    this.layer = layer;
  }
  
  /**
   * Plot plot canvas.
   *
   * @param title    the title
   * @param plotData the plot data
   * @param function the function
   * @return the plot canvas
   */
  public static PlotCanvas plot(String title, List<double[]> plotData, Function<double[], double[]> function) {
    double[][] data = plotData.stream().map(function).toArray(i -> new double[i][]);
    return plot(title, data);
  }
  
  /**
   * Plot plot canvas.
   *
   * @param title the title
   * @param data  the data
   * @return the plot canvas
   */
  public static PlotCanvas plot(String title, double[][] data) {
    PlotCanvas plot = ScatterPlot.plot(data);
    plot.setTitle(title);
    plot.setAxisLabels("x", "y");
    plot.setSize(600, 400);
    return plot;
  }
  
  @Override
  public NNLayer getLayer() {
    return layer;
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {2, 3, 1}
    };
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {100, 100, 1}
    };
  }
  
  @Override
  public void test(NotebookOutput log) {
    super.test(log);
    
    log.h3("Function Plots");
    NNLayer layer = getLayer();
    List<double[]> plotData = scan().mapToObj(x -> {
      SimpleEval eval = SimpleEval.run(layer, new Tensor(x));
      return new double[]{x, eval.getOutput().get(0), eval.getDerivative()[0].get(0)};
    }).collect(Collectors.toList());
    
    log.code(() -> {
      return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
    });
    
    log.code(() -> {
      return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
    });
    
  }
  
  /**
   * Scan double stream.
   *
   * @return the double stream
   */
  public DoubleStream scan() {
    return IntStream.range(-1000, 1000).mapToDouble(x -> x / 300.0);
  }
  
}
