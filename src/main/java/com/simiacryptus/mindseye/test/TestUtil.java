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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.StochasticComponent;
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
import com.simiacryptus.util.io.GifSequenceWriter;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import guru.nidi.graphviz.attribute.RankDir;
import guru.nidi.graphviz.model.Factory;
import guru.nidi.graphviz.model.Graph;
import guru.nidi.graphviz.model.Link;
import guru.nidi.graphviz.model.LinkSource;
import guru.nidi.graphviz.model.LinkTarget;
import guru.nidi.graphviz.model.MutableNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.ref.WeakReference;
import java.net.URI;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Comparator;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.List;
import java.util.LongSummaryStatistics;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.IntToLongFunction;
import java.util.function.IntUnaryOperator;
import java.util.function.LongToIntFunction;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image apply util.
 */
public class TestUtil {
  /**
   * The constant S3_ROOT.
   */
  public static final URI S3_ROOT = URI.create("https://s3-us-west-2.amazonaws.com/simiacryptus/");
  private static final Logger log = LoggerFactory.getLogger(TestUtil.class);
  /**
   * The constant scheduledThreadPool.
   */
  public static ScheduledExecutorService scheduledThreadPool = Executors.newScheduledThreadPool(1);
  
  /**
   * Add logging.
   *
   * @param network the network
   */
  public static void addLogging(@Nonnull final DAGNetwork network) {
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
  public static void addMonitoring(@Nonnull final DAGNetwork network, @Nonnull final MonitoredObject monitoringRoot) {
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
  public static PlotCanvas compare(final String title, @Nonnull final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics xStatistics = Arrays.stream(trials)
        .flatMapToDouble(x -> x.history.stream().mapToDouble(step -> step.iteration))
        .filter(Double::isFinite)
        .summaryStatistics();
      final DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
        .flatMapToDouble(x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> Math.log10(step.fitness)))
        .filter(Double::isFinite)
        .summaryStatistics();
      if (xStatistics.getCount() == 0) {
        log.info("No Data");
        return null;
      }
      @Nonnull final double[] lowerBound = {xStatistics.getCount() == 0 ? 0 : xStatistics.getMin(), yStatistics.getCount() < 2 ? 0 : yStatistics.getMin()};
      @Nonnull final double[] upperBound = {xStatistics.getCount() == 0 ? 1 : xStatistics.getMax(), yStatistics.getCount() < 2 ? 1 : yStatistics.getMax()};
      @Nonnull final PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
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
      for (@Nonnull final ProblemRun trial : filtered) {
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
    } catch (@Nonnull final Exception e) {
      e.printStackTrace(System.out);
      return null;
    }
  }
  
  /**
   * To string string.
   *
   * @param fn the fn
   * @return the string
   */
  public static String toString(@Nonnull Consumer<PrintStream> fn) {
    @Nonnull ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    try (@Nonnull PrintStream out = new PrintStream(buffer)) {
      fn.accept(out);
    }
    return new String(buffer.toByteArray(), Charset.forName("UTF-8"));
  }
  
  /**
   * Compare plot canvas.
   *
   * @param title  the title
   * @param trials the trials
   * @return the plot canvas
   */
  public static PlotCanvas compareTime(final String title, @Nonnull final ProblemRun... trials) {
    try {
      final DoubleSummaryStatistics[] xStatistics = Arrays.stream(trials)
        .map(x -> x.history.stream().mapToDouble(step -> step.epochTime)
          .filter(Double::isFinite)
          .summaryStatistics()).toArray(i -> new DoubleSummaryStatistics[i]);
      final double totalTime = Arrays.stream(xStatistics).mapToDouble(x -> x.getMax() - x.getMin()).max().getAsDouble();
      final DoubleSummaryStatistics yStatistics = Arrays.stream(trials)
        .flatMapToDouble(x -> x.history.stream().filter(y -> y.fitness > 0).mapToDouble(step -> Math.log10(step.fitness)))
        .filter(Double::isFinite)
        .summaryStatistics();
      if (yStatistics.getCount() == 0) {
        log.info("No Data");
        return null;
      }
      @Nonnull final double[] lowerBound = {0, yStatistics.getCount() == 0 ? 0 : yStatistics.getMin()};
      @Nonnull final double[] upperBound = {totalTime / 1000.0, yStatistics.getCount() == 1 ? 0 : yStatistics.getMax()};
      @Nonnull final PlotCanvas canvas = new PlotCanvas(lowerBound, upperBound);
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
    } catch (@Nonnull final Exception e) {
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
  public static void extractPerformance(@Nonnull final NotebookOutput log, @Nonnull final DAGNetwork network) {
    log.p("Per-layer Performance Metrics:");
    log.code(() -> {
      @Nonnull final Map<CharSequence, MonitoringWrapperLayer> metrics = new HashMap<>();
      network.visitNodes(node -> {
        if (node.getLayer() instanceof MonitoringWrapperLayer) {
          @Nullable final MonitoringWrapperLayer layer = node.getLayer();
          Layer inner = layer.getInner();
          String str = inner.toString();
          str += " class=" + inner.getClass().getName();
//          if(inner instanceof MultiPrecision<?>) {
//            str += "; precision=" + ((MultiPrecision) inner).getPrecision().name();
//          }
          metrics.put(str, layer);
        }
      });
      TestUtil.log.info("Performance: \n\t" + metrics.entrySet().stream().sorted(Comparator.comparing(x -> -x.getValue().getForwardPerformance().getMean())).map(e -> {
        @Nonnull final PercentileStatistics performanceF = e.getValue().getForwardPerformance();
        @Nonnull final PercentileStatistics performanceB = e.getValue().getBackwardPerformance();
        return String.format("%.6fs +- %.6fs (%d) <- %s", performanceF.getMean(), performanceF.getStdDev(), performanceF.getCount(), e.getKey()) +
          (performanceB.getCount() == 0 ? "" : String.format("%n\tBack: %.6fs +- %.6fs (%s)", performanceB.getMean(), performanceB.getStdDev(), performanceB.getCount()));
      }).reduce((a, b) -> a + "\n\t" + b).get());
    });
    removeInstrumentation(network);
  }
  
  /**
   * Remove instrumentation.
   *
   * @param network the network
   */
  public static void removeInstrumentation(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(node.<MonitoringWrapperLayer>getLayer().getInner());
      }
    });
  }
  
  /**
   * Sample performance map.
   *
   * @param network the network
   * @return the map
   */
  public static Map<CharSequence, Object> samplePerformance(@Nonnull final DAGNetwork network) {
    @Nonnull final Map<CharSequence, Object> metrics = new HashMap<>();
    network.visitLayers(layer -> {
      if (layer instanceof MonitoringWrapperLayer) {
        MonitoringWrapperLayer monitoringWrapperLayer = (MonitoringWrapperLayer) layer;
        Layer inner = monitoringWrapperLayer.getInner();
        String str = inner.toString();
        str += " class=" + inner.getClass().getName();
        HashMap<CharSequence, Object> row = new HashMap<>();
        row.put("fwd", monitoringWrapperLayer.getForwardPerformance().getMetrics());
        row.put("rev", monitoringWrapperLayer.getBackwardPerformance().getMetrics());
        metrics.put(str, row);
      }
    });
    return metrics;
  }
  
  /**
   * Gets monitor.
   *
   * @param history the history
   * @return the monitor
   */
  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history) {return getMonitor(history, null);}
  
  /**
   * Gets monitor.
   *
   * @param history the history
   * @param network the network
   * @return the monitor
   */
  public static TrainingMonitor getMonitor(@Nonnull final List<StepRecord> history, final Layer network) {
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
      public void onStepComplete(@Nonnull final Step currentPoint) {
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
        if (null != network) {
          if (network instanceof StochasticComponent) {
            ((StochasticComponent) network).shuffle(System.nanoTime());
          }
          if (network instanceof DAGNetwork) {
            ((DAGNetwork) network).visitLayers(layer -> {
              if (layer instanceof StochasticComponent) {
                ((StochasticComponent) layer).shuffle(System.nanoTime());
              }
            });
          }
        }
        super.onStepComplete(currentPoint);
      }
    };
  }
  
  /**
   * Add performance wrappers.
   *
   * @param network the network
   */
  public static void instrumentPerformance(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      Layer layer = node.getLayer();
      if (layer instanceof MonitoringWrapperLayer) {
        ((MonitoringWrapperLayer) layer).shouldRecordSignalMetrics(false);
      }
      else {
        @Nonnull MonitoringWrapperLayer monitoringWrapperLayer = new MonitoringWrapperLayer(layer).shouldRecordSignalMetrics(false);
        node.setLayer(monitoringWrapperLayer);
        monitoringWrapperLayer.freeRef();
      }
    });
  }
  
  /**
   * Plot plot canvas.
   *
   * @param history the history
   * @return the plot canvas
   */
  public static PlotCanvas plot(@Nonnull final List<StepRecord> history) {
    try {
      final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      double[][] data = history.stream().map(step -> new double[]{
        step.iteration, Math.log10(Math.max(valueStats.getMin(), step.fitness))})
        .filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
        .toArray(i -> new double[i][]);
      if (Arrays.stream(data).mapToInt(x -> x.length).sum() == 0) return null;
      @Nonnull final PlotCanvas plot = ScatterPlot.plot(data);
      plot.setTitle("Convergence Plot");
      plot.setAxisLabels("Iteration", "log10(Fitness)");
      plot.setSize(600, 400);
      return plot;
    } catch (@Nonnull final Exception e) {
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
  public static PlotCanvas plotTime(@Nonnull final List<StepRecord> history) {
    try {
      final LongSummaryStatistics timeStats = history.stream().mapToLong(x -> x.epochTime).summaryStatistics();
      final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
      @Nonnull final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{
        (step.epochTime - timeStats.getMin()) / 1000.0, Math.log10(Math.max(valueStats.getMin(), step.fitness))})
        .filter(x -> Arrays.stream(x).allMatch(Double::isFinite))
        .toArray(i -> new double[i][]));
      plot.setTitle("Convergence Plot");
      plot.setAxisLabels("Time", "log10(Fitness)");
      plot.setSize(600, 400);
      return plot;
    } catch (@Nonnull final Exception e) {
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
  public static void printDataStatistics(@Nonnull final NotebookOutput log, @Nonnull final Tensor[][] data) {
    for (int col = 1; col < data[0].length; col++) {
      final int c = col;
      log.out("Learned Representation Statistics for Column " + col + " (all bands)");
      log.code(() -> {
        @Nonnull final ScalarStatistics scalarStatistics = new ScalarStatistics();
        Arrays.stream(data)
          .flatMapToDouble(row -> Arrays.stream(row[c].getData()))
          .forEach(v -> scalarStatistics.add(v));
        return scalarStatistics.getMetrics();
      });
      final int _col = col;
      log.out("Learned Representation Statistics for Column " + col + " (by band)");
      log.code(() -> {
        @Nonnull final int[] dimensions = data[0][_col].getDimensions();
        return IntStream.range(0, dimensions[2]).mapToObj(x -> x).flatMap(b -> {
          return Arrays.stream(data).map(r -> r[_col]).map(tensor -> {
            @Nonnull final ScalarStatistics scalarStatistics = new ScalarStatistics();
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
  public static void printHistory(@Nonnull final NotebookOutput log, @Nonnull final List<StepRecord> history) {
    if (!history.isEmpty()) {
      log.out("Convergence Plot: ");
      log.code(() -> {
        final DoubleSummaryStatistics valueStats = history.stream().mapToDouble(x -> x.fitness).filter(x -> x > 0).summaryStatistics();
        @Nonnull final PlotCanvas plot = ScatterPlot.plot(history.stream().map(step ->
          new double[]{step.iteration, Math.log10(Math.max(valueStats.getMin(), step.fitness))})
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
  public static void removeLogging(@Nonnull final DAGNetwork network) {
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
  public static void removeMonitoring(@Nonnull final DAGNetwork network) {
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
  public static CharSequence render(@Nonnull final NotebookOutput log, @Nonnull final Tensor tensor, final boolean normalize) {
    return TestUtil.renderToImages(tensor, normalize).map(image -> {
      return log.image(image, "");
    }).reduce((a, b) -> a + b).get();
  }
  
  /**
   * Render to images stream.
   *
   * @param tensor    the tensor
   * @param normalize the normalize
   * @return the stream
   */
  public static Stream<BufferedImage> renderToImages(@Nonnull final Tensor tensor, final boolean normalize) {
    final DoubleStatistics[] statistics = IntStream.range(0, tensor.getDimensions()[2]).mapToObj(band -> {
      return new DoubleStatistics().accept(tensor.coordStream(false)
        .filter(x -> x.getCoords()[2] == band)
        .mapToDouble(c -> tensor.get(c)).toArray());
    }).toArray(i -> new DoubleStatistics[i]);
    @Nonnull final BiFunction<Double, DoubleStatistics, Double> transform = (value, stats) -> {
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
    tensor.coordStream(true).collect(Collectors.groupingBy(x -> x.getCoords()[2], Collectors.toList()));
    @Nullable final Tensor normal = tensor.mapCoords((c) -> transform.apply(tensor.get(c), statistics[c.getCoords()[2]]))
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
  @Nonnull
  public static BufferedImage resize(@Nonnull final BufferedImage source, final int size) {return resize(source, size, false);}
  
  /**
   * Resize buffered image.
   *
   * @param source         the source
   * @param size           the size
   * @param preserveAspect the preserve aspect
   * @return the buffered image
   */
  @Nonnull
  public static BufferedImage resize(@Nonnull final BufferedImage source, final int size, boolean preserveAspect) {
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
  @Nonnull
  public static BufferedImage resize(BufferedImage source, int width, int height) {
    @Nonnull final BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
    @Nonnull final Graphics2D graphics = (Graphics2D) image.getGraphics();
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
  public static CharSequence toFormattedJson(final Object metrics) {
    try {
      @Nonnull final ByteArrayOutputStream out = new ByteArrayOutputStream();
      JsonUtil.writeJson(out, metrics);
      return out.toString();
    } catch (@Nonnull final IOException e1) {
      throw new RuntimeException(e1);
    }
  }
  
  /**
   * To graph graph.
   *
   * @param network the network
   * @return the graph
   */
  public static Graph toGraph(@Nonnull final DAGNetwork network) {
    final List<DAGNode> nodes = network.getNodes();
    final Map<UUID, MutableNode> graphNodes = nodes.stream().collect(Collectors.toMap(node -> node.getId(), node -> {
      @Nullable String name;
      @Nullable final Layer layer = node.getLayer();
      if (null == layer) {
        name = node.getId().toString();
      }
      else {
        final Class<? extends Layer> layerClass = layer.getClass();
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
  
  /**
   * Shuffle int stream.
   *
   * @param stream the stream
   * @return the int stream
   */
  public static IntStream shuffle(@Nonnull IntStream stream) {
    // http://primes.utm.edu/lists/small/10000.txt
    long coprimeA = 41387;
    long coprimeB = 9967;
    long ringSize = coprimeA * coprimeB - 1;
    @Nonnull IntToLongFunction fn = x -> (x * coprimeA * coprimeA) % ringSize;
    @Nonnull LongToIntFunction inv = x -> (int) ((x * coprimeB * coprimeB) % ringSize);
    @Nonnull IntUnaryOperator conditions = x -> {
      assert x < ringSize;
      assert x >= 0;
      return x;
    };
    return stream.map(conditions).mapToLong(fn).sorted().mapToInt(inv);
  }
  
  /**
   * Run all.
   *
   * @param runnables the runnables
   */
  public static void runAllParallel(@Nonnull Runnable... runnables) {
    Arrays.stream(runnables)
      .parallel()
      .forEach(Runnable::run);
  }
  
  /**
   * Run all serial.
   *
   * @param runnables the runnables
   */
  public static void runAllSerial(@Nonnull Runnable... runnables) {
    Arrays.stream(runnables)
      .forEach(Runnable::run);
  }
  
  /**
   * Or else supplier.
   *
   * @param <T>       the type parameter
   * @param suppliers the suppliers
   * @return the supplier
   */
  public static <T> Supplier<T> orElse(@Nonnull Supplier<T>... suppliers) {
    return () -> {
      for (@Nonnull Supplier<T> supplier : suppliers) {
        T t = supplier.get();
        if (null != t) return t;
      }
      return null;
    };
  }
  
  /**
   * Mini stack trace string.
   *
   * @return the string
   */
  public static CharSequence miniStackTrace() {
    int max = 30;
    StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
    List<CharSequence> list = Arrays.stream(stackTrace).skip(3).limit(max - 3).map(x -> x.isNativeMethod() ? "(Native Method)" :
      (x.getFileName() != null && x.getLineNumber() >= 0 ?
        x.getFileName() + ":" + x.getLineNumber() :
        (x.getFileName() != null ? x.getFileName() : "(Unknown Source)"))).collect(Collectors.toList());
    return "[" + list.stream().reduce((a, b) -> a + ", " + b).get() + (stackTrace.length > max ? ", ..." : "") + "]";
  }
  
  /**
   * Monitor ui.
   *
   * @param input       the input
   * @param exitOnClose the exit on close
   * @param normalize   the normalize
   */
  public static void monitorImage(final Tensor input, final boolean exitOnClose, final boolean normalize) {monitorImage(input, exitOnClose, 30, normalize);}
  
  /**
   * Monitor ui.
   *
   * @param input       the input
   * @param exitOnClose the exit on close
   * @param period      the period
   * @param normalize   the normalize
   */
  public static void monitorImage(final Tensor input, final boolean exitOnClose, final int period, final boolean normalize) {
    if (GraphicsEnvironment.isHeadless() || !Desktop.isDesktopSupported() || !Desktop.getDesktop().isSupported(Desktop.Action.BROWSE))
      return;
    JLabel label = new JLabel(new ImageIcon(input.toImage()));
    WeakReference<JLabel> labelWeakReference = new WeakReference<>(label);
    ScheduledFuture<?> updater = scheduledThreadPool.scheduleAtFixedRate(() -> {
      try {
        JLabel jLabel = labelWeakReference.get();
        if (null != jLabel) {
          BufferedImage image = (normalize ? normalizeBands(input) : input).toImage();
          int width = jLabel.getWidth();
          if (width > 0) TestUtil.resize(image, width, jLabel.getHeight());
          jLabel.setIcon(new ImageIcon(image));
        }
      } catch (Throwable e) {
        log.warn("Error updating image", e);
      }
    }, 0, period, TimeUnit.SECONDS);
    new Thread(() -> {
      final JDialog dialog;
      Window window = JOptionPane.getRootFrame();
      String title = "Image: " + Arrays.toString(input.getDimensions());
      if (window instanceof Frame) {
        dialog = new JDialog((Frame) window, title, true);
      }
      else {
        dialog = new JDialog((Dialog) window, title, true);
      }
      dialog.setResizable(false);
      dialog.setSize(input.getDimensions()[0], input.getDimensions()[1]);
      JMenuBar menu = new JMenuBar();
      JMenu fileMenu = new JMenu("File");
      JMenuItem saveAction = new JMenuItem("Save");
      fileMenu.add(saveAction);
      saveAction.addActionListener(new ActionListener() {
        @Override
        public void actionPerformed(final ActionEvent e) {
          JFileChooser fileChooser = new JFileChooser();
          fileChooser.setAcceptAllFileFilterUsed(false);
          fileChooser.addChoosableFileFilter(new FileFilter() {
            @Override
            public boolean accept(final File f) {
              return f.getName().toUpperCase().endsWith(".PNG");
            }
  
            @Override
            public String getDescription() {
              return "*.png";
            }
          });
  
          int result = fileChooser.showSaveDialog(dialog);
          if (JFileChooser.APPROVE_OPTION == result) {
            try {
              File selectedFile = fileChooser.getSelectedFile();
              if (!selectedFile.getName().toUpperCase().endsWith(".PNG"))
                selectedFile = new File(selectedFile.getParent(), selectedFile.getName() + ".png");
              BufferedImage image = (normalize ? normalizeBands(input) : input).toImage();
              if (!ImageIO.write(image, "PNG", selectedFile)) throw new IllegalArgumentException();
            } catch (IOException e1) {
              throw new RuntimeException(e1);
            }
          }
        }
      });
      menu.add(fileMenu);
      dialog.setJMenuBar(menu);
      
      Container contentPane = dialog.getContentPane();
      contentPane.setLayout(new BorderLayout());
      contentPane.add(label, BorderLayout.CENTER);
      //contentPane.add(dialog, BorderLayout.CENTER);
      if (JDialog.isDefaultLookAndFeelDecorated()) {
        boolean supportsWindowDecorations = UIManager.getLookAndFeel().getSupportsWindowDecorations();
        if (supportsWindowDecorations) {
          dialog.setUndecorated(true);
          SwingUtilities.getRootPane(dialog).setWindowDecorationStyle(JRootPane.PLAIN_DIALOG);
        }
      }
      dialog.pack();
      dialog.setLocationRelativeTo(null);
      dialog.addComponentListener(new ComponentAdapter() {
        @Override
        public void componentResized(final ComponentEvent e) {
          //dialog.pack();
          super.componentResized(e);
          BufferedImage image = input.toImage();
          int width = e.getComponent().getWidth();
          if (width > 0) TestUtil.resize(image, width, e.getComponent().getHeight());
          label.setIcon(new ImageIcon(image));
          dialog.pack();
        }
      });
      dialog.addWindowListener(new WindowAdapter() {
        private boolean gotFocus = false;
        
        public void windowClosed(WindowEvent e) {
          dialog.getContentPane().removeAll();
          updater.cancel(false);
          if (exitOnClose) System.exit(0);
        }
        
        public void windowGainedFocus(WindowEvent we) {
          // Once window gets focus, set initial focus
          if (!gotFocus) {
            gotFocus = true;
          }
        }
        
      });
      dialog.show();
      dialog.dispose();
    }).start();
  }
  
  /**
   * Normalize bands tensor.
   *
   * @param image the image
   * @return the tensor
   */
  public static Tensor normalizeBands(final Tensor image) {return normalizeBands(image, 255);}
  
  /**
   * Normalize bands tensor.
   *
   * @param image the image
   * @param max   the max
   * @return the tensor
   */
  public static Tensor normalizeBands(final Tensor image, final int max) {
    DoubleStatistics[] statistics = IntStream.range(0, image.getDimensions()[2]).mapToObj(i -> new DoubleStatistics()).toArray(i -> new DoubleStatistics[i]);
    image.coordStream(false).forEach(c -> {
      double value = image.get(c);
      statistics[c.getCoords()[2]].accept(value);
    });
    return image.mapCoords(c -> {
      double value = image.get(c);
      DoubleStatistics statistic = statistics[c.getCoords()[2]];
      return max * (value - statistic.getMin()) / (statistic.getMax() - statistic.getMin());
    });
  }
  
  /**
   * To string string.
   *
   * @param stack the stack
   * @return the string
   */
  public static String toString(final StackTraceElement[] stack) {return toString(stack, "\n");}
  
  /**
   * To string string.
   *
   * @param stack     the stack
   * @param delimiter the delimiter
   * @return the string
   */
  public static String toString(final StackTraceElement[] stack, final CharSequence delimiter) {
    return Arrays.stream(stack).map(x -> x.getFileName() + ":" + x.getLineNumber()).reduce((a, b) -> a + delimiter + b).orElse("");
  }
  
  /**
   * Gets caller.
   *
   * @return the caller
   */
  public static CharSequence getCaller() {
    return toString(getStackTrace(4));
  }
  
  /**
   * Get stack trace stack trace element [ ].
   *
   * @return the stack trace element [ ]
   */
  public static StackTraceElement[] getStackTrace() {return getStackTrace(5);}
  
  /**
   * Get stack trace stack trace element [ ].
   *
   * @param skip the skip
   * @return the stack trace element [ ]
   */
  public static StackTraceElement[] getStackTrace(final int skip) {
    StackTraceElement[] elements = Arrays.stream(Thread.currentThread().getStackTrace()).skip(skip)
      .filter(x -> x.getClassName().startsWith("com.simiacryptus.mindseye.")
        && !x.getClassName().startsWith("com.simiacryptus.mindseye.lang.")
        && !x.getClassName().startsWith("com.simiacryptus.mindseye.test."))
      .limit(1)
      .toArray(i -> new StackTraceElement[i]);
    if (0 == elements.length) {
      elements = Arrays.stream(Thread.currentThread().getStackTrace()).skip(skip)
        .filter(x -> x.getClassName().startsWith("com.simiacryptus.mindseye."))
        .limit(1)
        .toArray(i -> new StackTraceElement[i]);
    }
    return elements;
  }
  
  private static int gifNumber = 0;
  
  /**
   * Animated gif char sequence.
   *
   * @param log    the log
   * @param images the images
   * @return the char sequence
   */
  public static CharSequence animatedGif(@Nonnull final NotebookOutput log, @Nonnull final BufferedImage... images) {return animatedGif(log, 15000, images);}
  
  /**
   * Animated gif char sequence.
   *
   * @param log        the log
   * @param loopTimeMs the loop time ms
   * @param images     the images
   * @return the char sequence
   */
  public static CharSequence animatedGif(@Nonnull final NotebookOutput log, final int loopTimeMs, @Nonnull final BufferedImage... images) {
    try {
      @Nonnull String filename = gifNumber++ + ".gif";
      @Nonnull File file = new File(log.getResourceDir(), filename);
      GifSequenceWriter.write(file, loopTimeMs / images.length, true, images);
      return String.format("<img src=\"etc/%s\" />", filename);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  public static void writeGif(@Nonnull final NotebookOutput log, final Stream<BufferedImage> imageStream) {
    BufferedImage[] imgs = imageStream.toArray(i -> new BufferedImage[i]);
    log.p("Animated Sequence:");
    log.p(animatedGif(log, imgs));
  }
  
  @Nonnull
  public static <K, V> Map<K, V> map(Consumer<Map<K, V>> configure) {
    Map<K, V> map = new HashMap<>();
    configure.accept(map);
    return map;
  }
  
  @Nonnull
  public static Supplier<DoubleStream> geometricStream(final double start, final double end, final int steps) {
    double step = Math.pow(end / start, 1.0 / steps);
    return () -> DoubleStream.iterate(start, x -> x * step).limit(steps);
  }
  
  @Nonnull
  public static Supplier<DoubleStream> arithmeticStream(final double start, final double end, final int steps) {
    double step = Math.pow(end - start, 1.0 / steps);
    return () -> DoubleStream.iterate(start, x -> x + step).limit(steps);
  }
  
  @Nonnull
  public static Supplier<DoubleStream> constantStream(final double... values) {
    return () -> Arrays.stream(values);
  }
  
  public static void browse(final URI uri) throws IOException {
    if (!GraphicsEnvironment.isHeadless() && Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE))
      Desktop.getDesktop().browse(uri);
  }
}
