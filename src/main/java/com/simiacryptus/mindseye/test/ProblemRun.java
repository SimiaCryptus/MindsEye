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

package com.simiacryptus.mindseye.test;

import smile.plot.LinePlot;
import smile.plot.Plot;
import smile.plot.ScatterPlot;

import java.awt.*;
import java.util.List;

/**
 * The type Problem eval.
 */
public class ProblemRun {
  
  /**
   * The Color.
   */
  public final Color color;
  /**
   * The History.
   */
  public final List<StepRecord> history;
  /**
   * The Name.
   */
  public final String name;
  /**
   * The Type.
   */
  public final PlotType type;
  
  /**
   * Instantiates a new Problem eval.
   *
   * @param name    the name
   * @param history the history
   * @param color   the color
   * @param type    the type
   */
  public ProblemRun(final String name, final List<StepRecord> history, final Color color, final PlotType type) {
    this.history = history;
    this.name = name;
    this.color = color;
    this.type = type;
  }
  
  /**
   * Plot plot.
   *
   * @param pts the pts
   * @return the plot
   */
  public Plot plot(final double[][] pts) {
    Plot plot;
    switch (type) {
      case Scatter:
        plot = new ScatterPlot(pts);
        plot.setID(name);
        plot.setColor(color);
        return plot;
      case Line:
        plot = new LinePlot(pts);
        plot.setID(name);
        plot.setColor(color);
        return plot;
      default:
        throw new IllegalStateException(type.toString());
    }
  }
  
  /**
   * The enum Plot type.
   */
  public enum PlotType {
    /**
     * Line plot type.
     */
    Line,
    /**
     * Scatter plot type.
     */
    Scatter
  }
}
