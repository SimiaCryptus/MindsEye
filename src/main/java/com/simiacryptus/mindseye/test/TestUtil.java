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

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.LoggingWrapperLayer;
import com.simiacryptus.mindseye.layers.java.MonitoringWrapperLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import guru.nidi.graphviz.attribute.RankDir;
import guru.nidi.graphviz.model.*;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image test util.
 */
public class TestUtil {
  
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
      step.iteraton, java.lang.Math.log10(step.fitness)})
      .filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
      .toArray(i -> new double[i][]));
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
      (step.epochTime - timeStats.getMin()) / 1000.0, java.lang.Math.log10(step.fitness)})
      .filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
      .toArray(i -> new double[i][]));
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
      .filter(Double::isFinite)
      .summaryStatistics();
    DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
      .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> java.lang.Math.log10(step.fitness)))
      .filter(Double::isFinite)
      .summaryStatistics();
    if (xStatistics.getCount() == 0) return null;
    double[] lowerBound = {xStatistics.getMin(), yStatistics.getMin()};
    double[] upperBound = {xStatistics.getMax(), yStatistics.getMax()};
    PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
    canvas.setTitle("Convergence Plot");
    canvas.setAxisLabels("Iteration", "log10(Fitness)");
    canvas.setSize(600, 400);
    List<ProblemRun> filtered = Arrays.stream(trials).filter(x -> !x.history.isEmpty()).collect(Collectors.toList());
    if (filtered.isEmpty()) return null;
    for (ProblemRun trial : filtered) {
      ScatterPlot plot = new ScatterPlot(trial.history.stream().map(step -> new double[]{
        step.iteraton, Math.log10(step.fitness)})
        .filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
        .toArray(i -> new double[i][]));
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
    DoubleSummaryStatistics[] xStatistics = Arrays.stream(trials)
      .map(x -> x.history.stream().mapToDouble(step -> step.epochTime)
        .filter(Double::isFinite)
        .summaryStatistics()).toArray(i -> new DoubleSummaryStatistics[i]);
    double totalTime = Arrays.stream(xStatistics).mapToDouble(x -> x.getMax() - x.getMin()).max().getAsDouble();
    DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
      .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> java.lang.Math.log10(step.fitness)))
      .filter(Double::isFinite)
      .summaryStatistics();
    if (yStatistics.getCount() == 0) return null;
    double[] lowerBound = {0, yStatistics.getMin()};
    double[] upperBound = {(totalTime) / 1000.0, yStatistics.getMax()};
    PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
    canvas.setTitle("Convergence Plot");
    canvas.setAxisLabels("Time", "log10(Fitness)");
    canvas.setSize(600, 400);
    List<ProblemRun> filtered = Arrays.stream(trials).filter(x -> !x.history.isEmpty()).collect(Collectors.toList());
    if (filtered.isEmpty()) return null;
    for (int t = 0; t < filtered.size(); t++) {
      ProblemRun trial = filtered.get(t);
      DoubleSummaryStatistics trialStats = xStatistics[t];
      ScatterPlot plot = new ScatterPlot(trial.history.stream().map(step -> {
        return new double[]{(step.epochTime - trialStats.getMin()) / 1000.0, java.lang.Math.log10(step.fitness)};
      }).filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
        .toArray(i -> new double[i][]));
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
   * Remove performance wrappers.
   *
   * @param log     the log
   * @param network the network
   */
  public static void extractPerformance(NotebookOutput log, DAGNetwork network) {
    log.p("Per-layer Performance Metrics:");
    log.code(() -> {
      Map<NNLayer, MonitoringWrapperLayer> metrics = new HashMap<>();
      network.visitNodes(node -> {
        if ((node.getLayer() instanceof MonitoringWrapperLayer)) {
          MonitoringWrapperLayer layer = node.getLayer();
          metrics.put(layer.getInner(), layer);
        }
      });
      System.out.println("Forward Performance: \n\t" + metrics.entrySet().stream().map(e -> {
        PercentileStatistics performance = e.getValue().getForwardPerformance();
        return String.format("%s -> %.6fs +- %.6fs (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
      }).reduce((a, b) -> a + "\n\t" + b));
      System.out.println("Backward Performance: \n\t" + metrics.entrySet().stream().map(e -> {
        PercentileStatistics performance = e.getValue().getBackwardPerformance();
        return String.format("%s -> %.6fs +- %.6fs (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
      }).reduce((a, b) -> a + "\n\t" + b));
    });
    log.p("Removing performance wrappers");
    log.code(() -> {
      network.visitNodes(node -> {
        if (node.getLayer() instanceof MonitoringWrapperLayer) {
          node.setLayer(node.<MonitoringWrapperLayer>getLayer().getInner());
        }
      });
    });
  }
  
  /**
   * Add performance wrappers.
   *
   * @param log     the log
   * @param network the network
   */
  public static void instrumentPerformance(NotebookOutput log, DAGNetwork network) {
    log.p("Adding performance wrappers");
    log.code(() -> {
      network.visitNodes(node -> {
        if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
          node.setLayer(new MonitoringWrapperLayer(node.getLayer()).shouldRecordSignalMetrics(false));
        }
        else {
          ((MonitoringWrapperLayer) node.getLayer()).shouldRecordSignalMetrics(false);
        }
      });
    });
  }
  
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
   * Add logging.
   *
   * @param network the network
   */
  public static void addLogging(DAGNetwork network) {
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof LoggingWrapperLayer)) {
        node.setLayer(new LoggingWrapperLayer(node.getLayer()));
      }
    });
  }
  
  /**
   * Remove monitoring.
   *
   * @param network the network
   */
  public static void removeLogging(DAGNetwork network) {
    network.visitNodes(node -> {
      if (node.getLayer() instanceof LoggingWrapperLayer) {
        node.setLayer(((LoggingWrapperLayer) node.getLayer()).getInner());
      }
    });
  }
  
  /**
   * Print data statistics.
   *
   * @param log  the log
   * @param data the data
   */
  public static void printDataStatistics(NotebookOutput log, Tensor[][] data) {
    for (int col = 1; col < data[0].length; col++) {
      int c = col;
      log.out("Learned Representation Statistics for Column " + col + " (all bands)");
      log.code(() -> {
        ScalarStatistics scalarStatistics = new ScalarStatistics();
        Arrays.stream(data)
          .flatMapToDouble(row -> Arrays.stream(row[c].getData()))
          .forEach(v -> scalarStatistics.add(v));
        return scalarStatistics.getMetrics();
      });
      int _col = col;
      log.out("Learned Representation Statistics for Column " + col + " (by band)");
      log.code(() -> {
        int[] dimensions = data[0][_col].getDimensions();
        return IntStream.range(0, dimensions[2]).mapToObj(x -> x).flatMap(b -> {
          return Arrays.stream(data).map(r -> r[_col]).map(tensor -> {
            ScalarStatistics scalarStatistics = new ScalarStatistics();
            scalarStatistics.add(new Tensor(dimensions[0], dimensions[1]).fillByCoord(coord -> tensor.get(coord.getCoords()[0], coord.getCoords()[1], b)).getData());
            return scalarStatistics;
          });
        }).map(x -> x.getMetrics().toString()).reduce((a, b) -> a + "\n" + b).get();
      });
    }
  }
  
  /**
   * Print history.
   *
   * @param log     the log
   * @param history the history
   */
  public static void printHistory(NotebookOutput log, List<Step> history) {
    if (!history.isEmpty()) {
      log.out("Convergence Plot: ");
      log.code(() -> {
        PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())}).toArray(i -> new double[i][]));
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "log10(Fitness)");
        plot.setSize(600, 400);
        return plot;
      });
    }
  }
  
  /**
   * Render string.
   *
   * @param log       the log
   * @param tensor    the tensor
   * @param normalize the normalize
   * @return the string
   */
  public static String render(NotebookOutput log, Tensor tensor, boolean normalize) {
    return renderToImages(tensor, normalize).map(image -> {
      try {
        return log.image(image, "");
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }).reduce((a, b) -> a + b).get();
  }
  
  /**
   * Render to images stream.
   *
   * @param tensor    the tensor
   * @param normalize the normalize
   * @return the stream
   */
  public static Stream<BufferedImage> renderToImages(Tensor tensor, boolean normalize) {
    DoubleStatistics[] statistics = IntStream.range(0, tensor.getDimensions()[2]).mapToObj(band -> {
      return new DoubleStatistics().accept(tensor.coordStream()
        .filter(x -> x.getCoords()[2] == band)
        .mapToDouble(c -> tensor.get(c)).toArray());
    }).toArray(i -> new DoubleStatistics[i]);
    BiFunction<Double, DoubleStatistics, Double> transform = (value, stats) -> {
      double width = Math.sqrt(2) * stats.getStandardDeviation();
      double centered = value - stats.getAverage();
      double distance = Math.abs(value - stats.getAverage());
      double positiveMax = stats.getMax() - stats.getAverage();
      double negativeMax = stats.getAverage() - stats.getMin();
      final double unitValue;
      if (value < centered) {
        if (distance > width) {
          unitValue = 0.25 - 0.25 * ((distance - width) / (negativeMax - width));
        }
        else {
          unitValue = 0.5 - 0.25 * ((distance) / (width));
        }
      }
      else {
        if (distance > width) {
          unitValue = 0.75 + 0.25 * ((distance - width) / (positiveMax - width));
        }
        else {
          unitValue = 0.5 + 0.25 * ((distance) / (width));
        }
      }
      return (0xFF * unitValue);
    };
    tensor.coordStream().collect(Collectors.groupingBy(x -> x.getCoords()[2], Collectors.toList()));
    Tensor normal = tensor.mapCoords((c) -> transform.apply(tensor.get(c), statistics[c.getCoords()[2]]))
      .map(v -> Math.min(0xFF, Math.max(0, v)));
    return (normalize ? normal : tensor).toImages().stream();
  }
  
  /**
   * Resize buffered image.
   *
   * @param source the source
   * @param size   the size
   * @return the buffered image
   */
  public static BufferedImage resize(BufferedImage source, int size) {
    BufferedImage image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB);
    Graphics2D graphics = (Graphics2D) image.getGraphics();
    graphics.setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC));
    graphics.drawImage(source, 0, 0, size, size, null);
    return image;
  }
}
