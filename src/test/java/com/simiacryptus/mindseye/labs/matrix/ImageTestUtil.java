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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.layers.java.MonitoringWrapperLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.io.JsonUtil;
import smile.math.Math;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.awt.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.List;

/**
 * The type Image test util.
 */
public class ImageTestUtil {
  
  /**
   * The constant originalOut.
   */
  public static final PrintStream originalOut = System.out;
  
  /**
   * Gets monitor.
   *
   * @param history the history
   * @return the monitor
   */
  public static TrainingMonitor getMonitor(List<StepRecord> history) {
    return new TrainingMonitor() {
      @Override
      public void log(String msg) {
        System.out.println(msg);
        if (null != originalOut && System.out != originalOut) originalOut.println(msg);
        super.log(msg);
      }
      
      @Override
      public void onStepComplete(Step currentPoint) {
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
        super.onStepComplete(currentPoint);
      }
      
      @Override
      public void clear() {
        super.clear();
      }
    };
  }
  
  /**
   * Plot plot canvas.
   *
   * @param history the history
   * @return the plot canvas
   */
  public static PlotCanvas plot(List<StepRecord> history) {
    PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{
      step.iteraton, Math.log10(step.fitness)}).toArray(i -> new double[i][]));
    plot.setTitle("Convergence Plot");
    plot.setAxisLabels("Iteration", "log10(Fitness)");
    plot.setSize(600, 400);
    return plot;
  }
  
  /**
   * Plot plot canvas.
   *
   * @param history the history
   * @return the plot canvas
   */
  public static PlotCanvas plotTime(List<StepRecord> history) {
    LongSummaryStatistics timeStats = history.stream().mapToLong(x -> x.epochTime).summaryStatistics();
    PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{
      (step.iteraton-timeStats.getMin())/1000.0, Math.log10(step.fitness)}).toArray(i -> new double[i][]));
    plot.setTitle("Convergence Plot");
    plot.setAxisLabels("Time", "log10(Fitness)");
    plot.setSize(600, 400);
    return plot;
  }
  
  /**
   * Compare plot canvas.
   *
   * @param trials the trials
   * @return the plot canvas
   */
  public static PlotCanvas compare(ProblemRun... trials) {
    DoubleSummaryStatistics xStatistics = Arrays.stream(trials)
      .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> step.iteraton))
      .summaryStatistics();
    DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
      .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> Math.log10(step.fitness)))
      .summaryStatistics();
    double[] lowerBound = new double[]{xStatistics.getMin(), yStatistics.getMin()};
    double[] upperBound = new double[]{xStatistics.getMax(), yStatistics.getMax()};
    PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
    canvas.setTitle("Convergence Plot");
    canvas.setAxisLabels("Iteration", "log10(Fitness)");
    canvas.setSize(600, 400);
    for (ProblemRun trial : trials) {
      ScatterPlot plot = new ScatterPlot(trial.history.stream().map(step -> new double[]{
        step.iteraton, Math.log10(step.fitness)}).toArray(i -> new double[i][]));
      plot.setID(trial.name);
      plot.setColor(trial.color);
      canvas.add(plot);
    }
    return canvas;
  }
  
  /**
   * Compare plot canvas.
   *
   * @param trials the trials
   * @return the plot canvas
   */
  public static PlotCanvas compareTime(ProblemRun... trials) {
    DoubleSummaryStatistics xStatistics = Arrays.stream(trials)
      .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> step.epochTime))
      .summaryStatistics();
    DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
      .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> Math.log10(step.fitness)))
      .summaryStatistics();
    double[] lowerBound = new double[]{0, yStatistics.getMin()};
    double[] upperBound = new double[]{(xStatistics.getMax()-xStatistics.getMin())/1000.0, yStatistics.getMax()};
    PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
    canvas.setTitle("Convergence Plot");
    canvas.setAxisLabels("Time", "log10(Fitness)");
    canvas.setSize(600, 400);
    for (ProblemRun trial : trials) {
      ScatterPlot plot = new ScatterPlot(trial.history.stream().map(step -> new double[]{
        (step.epochTime-xStatistics.getMin()) / 1000.0, Math.log10(step.fitness)}).toArray(i -> new double[i][]));
      plot.setID(trial.name);
      plot.setColor(trial.color);
      canvas.add(plot);
    }
    return canvas;
  }
  
  /**
   * Add monitoring.
   *
   * @param network        the network
   * @param monitoringRoot the monitoring root
   */
  public static void addMonitoring(DAGNetwork network, MonitoredObject monitoringRoot) {
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).addTo(monitoringRoot));
      }
    });
  }
  
  /**
   * Remove monitoring.
   *
   * @param network the network
   */
  public static void removeMonitoring(DAGNetwork network) {
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(((MonitoringWrapperLayer) node.getLayer()).getInner());
      }
    });
  }
  
  /**
   * To formatted json string.
   *
   * @param metrics the metrics
   * @return the string
   */
  public static String toFormattedJson(Map<String, Object> metrics) {
    try {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      JsonUtil.writeJson(out, metrics);
      return out.toString();
    } catch (IOException e1) {
      throw new RuntimeException(e1);
    }
  }
  
  /**
   * The type Step record.
   */
  public static class StepRecord {
    /**
     * The Fitness.
     */
    public final double fitness;
    /**
     * The Epoch time.
     */
    public final long epochTime;
    /**
     * The Iteraton.
     */
    public final long iteraton;
  
    /**
     * Instantiates a new Step record.
     *
     * @param fitness   the fitness
     * @param epochTime the epoch time
     * @param iteraton  the iteraton
     */
    public StepRecord(double fitness, long epochTime, long iteraton) {
      this.fitness = fitness;
      this.epochTime = epochTime;
      this.iteraton = iteraton;
    }
  }
  
  /**
   * The type Problem run.
   */
  public static class ProblemRun {
    /**
     * The History.
     */
    public final List<StepRecord> history;
    /**
     * The Name.
     */
    public final String name;
    /**
     * The Color.
     */
    public final Color color;
  
    /**
     * Instantiates a new Problem run.
     *  @param name    the name
     * @param color   the color
     * @param history the history
     */
    public ProblemRun(String name, Color color, List<StepRecord> history) {
      this.history = history;
      this.name = name;
      this.color = color;
    }
  }
  
}
