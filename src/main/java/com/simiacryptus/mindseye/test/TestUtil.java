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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URI;
import java.util.*;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image run util.
 */
public class TestUtil {
  /**
   * The constant originalOut.
   */
  public static final PrintStream originalOut = System.out;
  /**
   * The constant S3_ROOT.
   */
  public static final URI S3_ROOT = URI.create("https://s3-us-west-2.amazonaws.com/simiacryptus/");
  private static final Logger log = LoggerFactory.getLogger(TestUtil.class);
  
  /**
   * Add logging.
   *
   * @param network the network
   */
  public static void addLogging(final DAGNetwork network) {
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof LoggingWrapperLayer)) {
        node.setLayer(new LoggingWrapperLayer(node.getLayer()));
      }
    });
  }
  
  /**
   * Add monitoring.
   *
   * @param network        the network
   * @param monitoringRoot the monitoring root
   */
  public static void addMonitoring(final DAGNetwork network, final MonitoredObject monitoringRoot) {
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).addTo(monitoringRoot));
      }
    });
  }
  
  /**
   * Compare plot canvas.
   *
   * @param title  the title
   * @param trials the trials
   * @return the plot canvas
   */
  public static PlotCanvas compare(final String title, final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics xStatistics = Arrays.stream(trials)
                                                        .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> step.iteration))
                                                        .filter(Double::isFinite)
                                                        .summaryStatistics();
      final DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
                                                        .flatMapToDouble(x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> java.lang.Math.log10(step.fitness)))
                                                        .filter(Double::isFinite)
                                                        .summaryStatistics();
      if (xStatistics.getCount() == 0) {
        log.info("No Data");
        return null;
      }
      final double[] lowerBound = {xStatistics.getCount() == 0 ? 0 : xStatistics.getMin(), yStatistics.getCount() < 2 ? 0 : yStatistics.getMin()};
      final double[] upperBound = {xStatistics.getCount() == 0 ? 1 : xStatistics.getMax(), yStatistics.getCount() < 2 ? 1 : yStatistics.getMax()};
      final PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
      canvas.setTitle(title);
      canvas.setAxisLabels("Iteration", "log10(Fitness)");
      canvas.setSize(600, 400);
      final List<ProblemRun> filtered = Arrays.stream(trials).filter(x -> !x.history.isEmpty()).collect(Collectors.toList());
      if (filtered.isEmpty()) {
        log.info("No Data");
        return null;
      }
      DoubleSummaryStatistics valueStatistics = filtered.stream().flatMap(x -> x.history.stream()).mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      log.info(String.format("Plotting range=%s, %s; valueStats=%s", Arrays.toString(lowerBound), Arrays.toString(upperBound), valueStatistics));
      for (final ProblemRun trial : filtered) {
        final double[][] pts = trial.history.stream().map(step -> new double[]{
          step.iteration, Math.log10(Math.max(step.fitness, valueStatistics.getMin()))})
                                            .filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
                                            .toArray(i -> new double[i][]);
        if (pts.length > 1) {
          log.info(String.format("Plotting %s points for %s", pts.length, trial.name));
          canvas.add(trial.plot(pts));
        }
        else {
          log.info(String.format("Only %s points for %s", pts.length, trial.name));
        }
      }
      return canvas;
    } catch (final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }
  
  /**
   * Compare plot canvas.
   *
   * @param title  the title
   * @param trials the trials
   * @return the plot canvas
   */
  public static PlotCanvas compareTime(final String title, final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics[] xStatistics = Arrays.stream(trials)
                                                          .map(x -> x.history.stream().mapToDouble(step -> step.epochTime)
                                                                             .filter(Double::isFinite)
                                                                             .summaryStatistics()).toArray(i -> new DoubleSummaryStatistics[i]);
      final double totalTime = Arrays.stream(xStatistics).mapToDouble(x -> x.getMax() - x.getMin()).max().getAsDouble();
      final DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
                                                        .flatMapToDouble(x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> java.lang.Math.log10(step.fitness)))
                                                        .filter(Double::isFinite)
                                                        .summaryStatistics();
      if (yStatistics.getCount() == 0) {
        log.info("No Data");
        return null;
      }
      final double[] lowerBound = {0, yStatistics.getCount() == 0 ? 0 : yStatistics.getMin()};
      final double[] upperBound = {totalTime / 1000.0, yStatistics.getCount() == 1 ? 0 : yStatistics.getMax()};
      final PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
      canvas.setTitle(title);
      canvas.setAxisLabels("Time", "log10(Fitness)");
      canvas.setSize(600, 400);
      final List<ProblemRun> filtered = Arrays.stream(trials).filter(x -> !x.history.isEmpty()).collect(Collectors.toList());
      if (filtered.isEmpty()) {
        log.info("No Data");
        return null;
      }
      DoubleSummaryStatistics valueStatistics = filtered.stream().flatMap(x -> x.history.stream()).mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      log.info(String.format("Plotting range=%s, %s; valueStats=%s", Arrays.toString(lowerBound), Arrays.toString(upperBound), valueStatistics));
      for (int t = 0; t < filtered.size(); t++) {
        final ProblemRun trial = filtered.get(t);
        final DoubleSummaryStatistics trialStats = xStatistics[t];
        final double[][] pts = trial.history.stream().map(step -> {
          return new double[]{(step.epochTime - trialStats.getMin()) / 1000.0, Math.log10(Math.max(step.fitness, valueStatistics.getMin()))};
        }).filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
                                            .toArray(i -> new double[i][]);
        if (pts.length > 1) {
          log.info(String.format("Plotting %s points for %s", pts.length, trial.name));
          canvas.add(trial.plot(pts));
        }
        else {
          log.info(String.format("Only %s points for %s", pts.length, trial.name));
        }
      }
      return canvas;
    } catch (final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }
  
  /**
   * Remove performance wrappers.
   *
   * @param log     the log
   * @param network the network
   */
  public static void extractPerformance(final NotebookOutput log, final DAGNetwork network) {
    log.p("Per-layer Performance Metrics:");
    log.code(() -> {
      final Map<NNLayer, MonitoringWrapperLayer> metrics = new HashMap<>();
      network.visitNodes(node -> {
        if (node.getLayer() instanceof MonitoringWrapperLayer) {
          final MonitoringWrapperLayer layer = node.getLayer();
          metrics.put(layer.getInner(), layer);
        }
      });
      TestUtil.log.info("Performance: \n\t" + metrics.entrySet().stream().map(e -> {
        final PercentileStatistics performanceF = e.getValue().getForwardPerformance();
        final PercentileStatistics performanceB = e.getValue().getBackwardPerformance();
        return String.format("%s -> %.6fs +- %.6fs (%d)", e.getKey(), performanceF.getMean(), performanceF.getStdDev(), performanceF.getCount()) +
          (performanceB.getCount() == 0 ? "" : String.format("%n\tBack: %.6fs +- %.6fs (%s)", performanceB.getMean(), performanceB.getStdDev(), performanceB.getCount()));
      }).reduce((a, b) -> a + "\n\t" + b).get());
    });
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(node.<MonitoringWrapperLayer>getLayer().getInner());
      }
    });
  }
  
  /**
   * Gets monitor.
   *
   * @param history the history
   * @return the monitor
   */
  public static TrainingMonitor getMonitor(final List<StepRecord> history) {
    return new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }
      
      @Override
      public void log(final String msg) {
        log.info(msg);
        super.log(msg);
      }
      
      @Override
      public void onStepComplete(final Step currentPoint) {
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
        super.onStepComplete(currentPoint);
      }
    };
  }
  
  /**
   * Add performance wrappers.
   *
   * @param log     the log
   * @param network the network
   */
  public static void instrumentPerformance(final NotebookOutput log, final DAGNetwork network) {
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).shouldRecordSignalMetrics(false));
      }
      else {
        ((MonitoringWrapperLayer) node.getLayer()).shouldRecordSignalMetrics(false);
      }
    });
  }
  
  /**
   * Plot plot canvas.
   *
   * @param history the history
   * @return the plot canvas
   */
  public static PlotCanvas plot(final List<StepRecord> history) {
    try {
      final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{
        step.iteration, java.lang.Math.log10(Math.max(valueStats.getMin(), step.fitness))})
                                                      .filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
                                                      .toArray(i -> new double[i][]));
      plot.setTitle("Convergence Plot");
      plot.setAxisLabels("Iteration", "log10(Fitness)");
      plot.setSize(600, 400);
      return plot;
    } catch (final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }
  
  /**
   * Plot plot canvas.
   *
   * @param history the history
   * @return the plot canvas
   */
  public static PlotCanvas plotTime(final List<StepRecord> history) {
    try {
      final LongSummaryStatistics timeStats = history.stream().mapToLong(x -> x.epochTime).summaryStatistics();
      final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{
        (step.epochTime - timeStats.getMin()) / 1000.0, java.lang.Math.log10(Math.max(valueStats.getMin(), step.fitness))})
                                                      .filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
                                                      .toArray(i -> new double[i][]));
      plot.setTitle("Convergence Plot");
      plot.setAxisLabels("Time", "log10(Fitness)");
      plot.setSize(600, 400);
      return plot;
    } catch (final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }
  
  /**
   * Print data statistics.
   *
   * @param log  the log
   * @param data the data
   */
  public static void printDataStatistics(final NotebookOutput log, final Tensor[][] data) {
    for (int col = 1; col < data[0].length; col++) {
      final int c = col;
      log.out("Learned Representation Statistics for Column " + col + " (all bands)");
      log.code(() -> {
        final ScalarStatistics scalarStatistics = new ScalarStatistics();
        Arrays.stream(data)
              .flatMapToDouble(row -> Arrays.stream(row[c].getData()))
              .forEach(v -> scalarStatistics.add(v));
        return scalarStatistics.getMetrics();
      });
      final int _col = col;
      log.out("Learned Representation Statistics for Column " + col + " (by band)");
      log.code(() -> {
        final int[] dimensions = data[0][_col].getDimensions();
        return IntStream.range(0, dimensions[2]).mapToObj(x -> x).flatMap(b -> {
          return Arrays.stream(data).map(r -> r[_col]).map(tensor -> {
            final ScalarStatistics scalarStatistics = new ScalarStatistics();
            scalarStatistics.add(new Tensor(dimensions[0], dimensions[1]).setByCoord(coord -> tensor.get(coord.getCoords()[0], coord.getCoords()[1], b)).getData());
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
  public static void printHistory(final NotebookOutput log, final List<StepRecord> history) {
    if (!history.isEmpty()) {
      log.out("Convergence Plot: ");
      log.code(() -> {
        final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
        final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step ->
                                                                        new double[]{step.iteration, java.lang.Math.log10(Math.max(valueStats.getMin(), step.fitness))})
                                                        .toArray(i -> new double[i][]));
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "log10(Fitness)");
        plot.setSize(600, 400);
        return plot;
      });
    }
  }
  
  /**
   * Remove monitoring.
   *
   * @param network the network
   */
  public static void removeLogging(final DAGNetwork network) {
    network.visitNodes(node -> {
      if (node.getLayer() instanceof LoggingWrapperLayer) {
        node.setLayer(((LoggingWrapperLayer) node.getLayer()).getInner());
      }
    });
  }
  
  /**
   * Remove monitoring.
   *
   * @param network the network
   */
  public static void removeMonitoring(final DAGNetwork network) {
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(((MonitoringWrapperLayer) node.getLayer()).getInner());
      }
    });
  }
  
  /**
   * Render string.
   *
   * @param log       the log
   * @param tensor    the tensor
   * @param normalize the normalize
   * @return the string
   */
  public static String render(final NotebookOutput log, final Tensor tensor, final boolean normalize) {
    return TestUtil.renderToImages(tensor, normalize).map(image -> {
      try {
        return log.image(image, "");
      } catch (final IOException e) {
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
  public static Stream<BufferedImage> renderToImages(final Tensor tensor, final boolean normalize) {
    final DoubleStatistics[] statistics = IntStream.range(0, tensor.getDimensions()[2]).mapToObj(band -> {
      return new DoubleStatistics().accept(tensor.coordStream()
                                                 .filter(x -> x.getCoords()[2] == band)
                                                 .mapToDouble(c -> tensor.get(c)).toArray());
    }).toArray(i -> new DoubleStatistics[i]);
    final BiFunction<Double, DoubleStatistics, Double> transform = (value, stats) -> {
      final double width = Math.sqrt(2) * stats.getStandardDeviation();
      final double centered = value - stats.getAverage();
      final double distance = Math.abs(value - stats.getAverage());
      final double positiveMax = stats.getMax() - stats.getAverage();
      final double negativeMax = stats.getAverage() - stats.getMin();
      final double unitValue;
      if (value < centered) {
        if (distance > width) {
          unitValue = 0.25 - 0.25 * ((distance - width) / (negativeMax - width));
        }
        else {
          unitValue = 0.5 - 0.25 * (distance / width);
        }
      }
      else {
        if (distance > width) {
          unitValue = 0.75 + 0.25 * ((distance - width) / (positiveMax - width));
        }
        else {
          unitValue = 0.5 + 0.25 * (distance / width);
        }
      }
      return 0xFF * unitValue;
    };
    tensor.coordStream().collect(Collectors.groupingBy(x -> x.getCoords()[2], Collectors.toList()));
    final Tensor normal = tensor.mapCoords((c) -> transform.apply(tensor.get(c), statistics[c.getCoords()[2]]))
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
  public static BufferedImage resize(final BufferedImage source, final int size) {return resize(source, size, false);}
  
  /**
   * Resize buffered image.
   *
   * @param source         the source
   * @param size           the size
   * @param preserveAspect the preserve aspect
   * @return the buffered image
   */
  public static BufferedImage resize(final BufferedImage source, final int size, boolean preserveAspect) {
    if (size < 0) return source;
    return resize(source, size, preserveAspect ? ((int) (size * (source.getHeight() * 1.0 / source.getWidth()))) : size);
  }
  
  /**
   * Resize buffered image.
   *
   * @param source the source
   * @param width  the width
   * @param height the height
   * @return the buffered image
   */
  public static BufferedImage resize(BufferedImage source, int width, int height) {
    final BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
    final Graphics2D graphics = (Graphics2D) image.getGraphics();
    graphics.setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC));
    graphics.drawImage(source, 0, 0, width, height, null);
    return image;
  }
  
  /**
   * To formatted json string.
   *
   * @param metrics the metrics
   * @return the string
   */
  public static String toFormattedJson(final Map<String, Object> metrics) {
    try {
      final ByteArrayOutputStream out = new ByteArrayOutputStream();
      JsonUtil.writeJson(out, metrics);
      return out.toString();
    } catch (final IOException e1) {
      throw new RuntimeException(e1);
    }
  }
  
  /**
   * To graph graph.
   *
   * @param network the network
   * @return the graph
   */
  public static Graph toGraph(final DAGNetwork network) {
    final List<DAGNode> nodes = network.getNodes();
    final Map<UUID, MutableNode> graphNodes = nodes.stream().collect(Collectors.toMap(node -> node.getId(), node -> {
      String name;
      final NNLayer layer = node.getLayer();
      if (null == layer) {
        name = node.getId().toString();
      }
      else {
        final Class<? extends NNLayer> layerClass = layer.getClass();
        name = layerClass.getSimpleName() + "\n" + layer.getId();
      }
      return Factory.mutNode(name);
    }));
    final Stream<UUID[]> stream = nodes.stream().flatMap(to -> {
      return Arrays.stream(to.getInputs()).map(from -> {
        return new UUID[]{from.getId(), to.getId()};
      });
    });
    final Map<UUID, List<UUID>> idMap = stream.collect(Collectors.groupingBy(x -> x[0],
                                                                             Collectors.mapping(x -> x[1], Collectors.toList())));
    nodes.forEach(to -> {
      graphNodes.get(to.getId()).addLink(
        idMap.getOrDefault(to.getId(), Arrays.asList()).stream().map(from -> {
          return Link.to(graphNodes.get(from));
        }).<LinkTarget>toArray(i -> new LinkTarget[i]));
    });
    final LinkSource[] nodeArray = graphNodes.values().stream().map(x -> (LinkSource) x).toArray(i -> new LinkSource[i]);
    return Factory.graph().with(nodeArray).generalAttr().with(RankDir.TOP_TO_BOTTOM).directed();
  }
  
}
