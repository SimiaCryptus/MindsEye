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

package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.mindseye.data.Caltech101;
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.PoolingLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.text.TableOutput;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.data.PercentileStatistics;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.SysOutInterceptor;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image encoding util.
 */
class TestUtil {
  /**
   * The constant out.
   */
  protected static PrintStream rawOut = SysOutInterceptor.INSTANCE.getInner();
  
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
   * Print model.
   *
   * @param log     the log
   * @param network the network
   * @param modelNo the model no
   */
  public static void printModel(NotebookOutput log, NNLayer network, final int modelNo) {
    log.out("Learned Model Statistics: ");
    log.code(() -> {
      ScalarStatistics scalarStatistics = new ScalarStatistics();
      network.state().stream().flatMapToDouble(x -> Arrays.stream(x))
        .forEach(v -> scalarStatistics.add(v));
      return scalarStatistics.getMetrics();
    });
    String modelName = "model" + modelNo + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));
  }
  
  /**
   * Validation report.
   *
   * @param log          the log
   * @param data         the data
   * @param dataPipeline the data pipeline
   * @param maxRows      the max rows
   */
  public static void validationReport(NotebookOutput log, Tensor[][] data, List<NNLayer> dataPipeline, int maxRows) {
    log.out("Current dataset and evaluation results: ");
    log.code(() -> {
      TableOutput table = new TableOutput();
      Arrays.stream(data).limit(maxRows).map(tensorArray -> {
        LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
        for (int col = 1; col < tensorArray.length; col++) {
          renderLayer(log, dataPipeline, row, col, tensorArray[col]);
        }
        return row;
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
  }
  
  /**
   * The constant svgNumber.
   */
  public static int svgNumber = 0;
  /**
   * The constant imageNumber.
   */
  public static int imageNumber = 0;
  
  /**
   * Render layer.
   *
   * @param log          the log
   * @param dataPipeline the data pipeline
   * @param row          the row
   * @param col          the col
   * @param tensor       the tensor
   */
  public static void renderLayer(NotebookOutput log, List<NNLayer> dataPipeline, LinkedHashMap<String, Object> row, int col, Tensor tensor) {
    row.put("Data_" + col, render(log, tensor, 0 < col));
    if (dataPipeline.size() >= col - 1 && 1 < col) {
      PipelineNetwork decoder = new PipelineNetwork();
      for (int i = col - 2; i >= 0; i--) {
        decoder.add(dataPipeline.get(i));
      }
      Tensor decoded = GpuController.call(ctx -> {
        return decoder.eval(ctx, tensor);
      }).getData().get(0);
      row.put("Decode_" + col, render(log, decoded, false));
      
      List<Tensor> rawComponents = IntStream.range(0, tensor.getDimensions()[2])
        .mapToObj(band -> findUnitComponent(decoder, band, tensor))
        .collect(Collectors.toList());
      Tensor baseline = findBaseline(decoder, tensor);
      List<Tensor> signedComponents = IntStream.range(0, tensor.getDimensions()[2])
        .mapToObj(band -> rawComponents.get(band).minus(baseline))
        .collect(Collectors.toList());
      
      row.put("SVG_" + col, log.file(toSvg(log, baseline, signedComponents), "svg" + svgNumber++ + ".svg", "SVG Composite Image"));
      
      String render = signedComponents.stream()
        .map(signedContribution -> render(log, signedContribution, true))
        .reduce((a, b) -> a + "" + b).get();
      row.put("Band_Decode_" + col, render);
    }
  }
  
  /**
   * To svg string.
   *
   * @param log              the log
   * @param baseline         the baseline
   * @param signedComponents the signed components
   * @return the string
   */
  public static String toSvg(NotebookOutput log, Tensor baseline, List<Tensor> signedComponents) {
    List<DoubleStatistics> componentStats = signedComponents.stream().map(t -> new DoubleStatistics().accept(t.getData())).collect(Collectors.toList());
    String positiveFilter = IntStream.range(0, signedComponents.size()).mapToObj(i -> {
      String name;
      try {
        name = String.format("component_%s.png", imageNumber++);
        ImageIO.write(signedComponents.get(i).map(v -> v > 0 ? (v*(0xFF/componentStats.get(i).getMax())) : 0).toImage(), "png", log.file(name));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      return String.format("  <feImage xlink:href=\"%s\" result=\"pos_image_%s\" />\n", name, i);
    }).reduce((a, b) -> a + "\n" + b).get();
    
    String negativeFilter = IntStream.range(0, signedComponents.size()).mapToObj(i -> {
      String name;
      try {
        name = String.format("component_%s.png", imageNumber++);
        ImageIO.write(signedComponents.get(i).map(v -> v < 0 ? (v*(0xFF/componentStats.get(i).getMin())) : 0).toImage(), "png", log.file(name));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      return String.format("  <feImage xlink:href=\"%s\" result=\"neg_image_%s\" />\n", name, i);
    }).reduce((a, b) -> a + "\n" + b).get();
    
    String compositingFilters = IntStream.range(0, signedComponents.size()).mapToObj(i ->
      "  <feComposite in=\"" + (i == 0 ? "FillPaint" : "lastResult") + "\" in2=\"neg_image_" + i + "\" result=\"lastResult\" operator=\"arithmetic\" k1=\"0.0\" k2=\"1.0\" k3=\""+(componentStats.get(i).getMin()/0xFF)+"\" k4=\"0.0\"/>\n" +
        "  <feComposite in=\"lastResult\" in2=\"pos_image_" + i + "\" result=\"lastResult\" operator=\"arithmetic\" k1=\"0.0\" k2=\"1.0\" k3=\""+(componentStats.get(i).getMax()/0xFF)+"\" k4=\"0.0\"/>\n").reduce((a, b) -> a + "\n" + b).get();
    
    int red = (int) baseline.get(0, 0, 0);
    int green = (int) baseline.get(0, 0, 1);
    int blue = (int) baseline.get(0, 0, 2);
    String avgHexColor = Long.toHexString(red + (green << 16) + (blue << 32));
    return "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n" +
      ("<defs>\n" +
        "<filter id=\"image\" >\n" + (
        positiveFilter + "\n" +
          negativeFilter + "\n" +
          compositingFilters
      ).replaceAll("\n", "\n\t") + "\n" +
        "</filter>\n" +
        "</defs>\n" +
        "<rect style=\"filter:url(#image);\" fill=\"#" + avgHexColor + "\" width=\"256\" height=\"256\"/>"
      ).replaceAll("\n", "\n\t") +
      "\n</svg>";
  }
  
  /**
   * Find baseline tensor.
   *
   * @param decoder the decoder
   * @param tensor  the tensor
   * @return the tensor
   */
  public static Tensor findBaseline(PipelineNetwork decoder, Tensor tensor) {
    PipelineNetwork decoderBand = new PipelineNetwork();
    double[] gate = new double[tensor.getDimensions()[2]];
    decoderBand.add(new ImgBandScaleLayer(gate));
    decoderBand.add(decoder);
    try {
      return GpuController.call(ctx -> {
        return decoder.eval(ctx, tensor.map(x -> 0));
      }).getData().get(0);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Find unit component tensor.
   *
   * @param decoder the decoder
   * @param band    the band
   * @param tensor  the tensor
   * @return the tensor
   */
  public static Tensor findUnitComponent(PipelineNetwork decoder, int band, Tensor tensor) {
    PipelineNetwork decoderBand = new PipelineNetwork();
    double[] gate = new double[tensor.getDimensions()[2]];
    gate[band] = tensor.getDimensions()[2];
    decoderBand.add(new ImgBandScaleLayer(gate));
    decoderBand.add(decoder);
    try {
      return GpuController.call(ctx -> {
        return decoderBand.eval(ctx, tensor);
      }).getData().get(0);
      //return log.image(t.toImage(), "");
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Add column tensor [ ] [ ].
   *
   * @param trainingData the training data
   * @param size         the size
   * @return the tensor [ ] [ ]
   */
  public static Tensor[][] addColumn(Tensor[][] trainingData, int... size) {
    return Arrays.stream(trainingData).map(x -> Stream.concat(
      Arrays.stream(x),
      Stream.of(new Tensor(size).fill(() -> 0.0 * (FastRandom.random() - 0.5))))
      .toArray(i -> new Tensor[i])).toArray(i -> new Tensor[i][]);
  }
  
  /**
   * Sets initial feature space.
   *
   * @param convolutionLayer the convolution layer
   * @param biasLayer        the bias layer
   * @param featureSpace     the feature space
   */
  public static void setInitialFeatureSpace(ConvolutionLayer convolutionLayer, ImgBandBiasLayer biasLayer, FindFeatureSpace featureSpace) {
    int[] filterDimensions = convolutionLayer.kernel.getDimensions();
    int outputBands = biasLayer.getBias().length;
    assert outputBands == biasLayer.getBias().length;
    int inputBands = filterDimensions[2] / outputBands;
    biasLayer.setWeights(i -> {
      double v = featureSpace.getAverages()[i];
      return Double.isFinite(v) ? v : biasLayer.getBias()[i];
    });
    Tensor[] featureSpaceVectors = featureSpace.getVectors();
    for (Tensor t : featureSpaceVectors) System.out.println(String.format("Feature Vector %s%n", t.prettyPrint()));
    convolutionLayer.kernel.fillByCoord(c -> {
      int kband = c.coords[2];
      int outband = kband % outputBands;
      int inband = (kband - outband) / outputBands;
      assert outband < outputBands;
      assert inband < inputBands;
      int x = c.coords[0];
      int y = c.coords[1];
      x = filterDimensions[0] - (x + 1);
      y = filterDimensions[1] - (y + 1);
      double v = featureSpaceVectors[inband].get(x, y, outband);
      return Double.isFinite(v) ? v : convolutionLayer.kernel.get(c);
    });
    System.out.println(String.format("Bias: %s%n", Arrays.toString(biasLayer.getBias())));
    System.out.println(String.format("Kernel: %s%n", convolutionLayer.kernel.prettyPrint()));
  }
  
  /**
   * Convolution features tensor [ ] [ ].
   *
   * @param tensors the tensors
   * @param radius  the radius
   * @return the tensor [ ] [ ]
   */
  public static Tensor[][] convolutionFeatures(Stream<Tensor[]> tensors, int radius) {
    return convolutionFeatures(tensors, radius, Math.max(3, radius));
  }
  
  /**
   * Convolution features tensor [ ] [ ].
   *
   * @param tensors the tensors
   * @param radius  the radius
   * @param padding the padding
   * @return the tensor [ ] [ ]
   */
  public static Tensor[][] convolutionFeatures(Stream<Tensor[]> tensors, int radius, int padding) {
    int column = 1;
    return tensors.parallel().flatMap(image -> {
      return IntStream.range(0, image[column].getDimensions()[0] - (radius - 1)).filter(x -> 1 == radius || 0 == x % padding).mapToObj(x -> x).flatMap(x -> {
        return IntStream.range(0, image[column].getDimensions()[column] - (radius - 1)).filter(y -> 1 == radius || 0 == y % padding).mapToObj(y -> {
          Tensor region = new Tensor(radius, radius, image[column].getDimensions()[2]);
          final ToDoubleBiFunction<Double, Coordinate> f = (v, c) -> {
            return image[column].get(c.coords[0] + x, c.coords[column] + y, c.coords[2]);
          };
          return new Tensor[]{image[0], region.mapCoords(f)};
        });
      });
    }).toArray(i -> new Tensor[i][]);
  }
  
  /**
   * Down stack tensors stream.
   *
   * @param stream the stream
   * @param factor the factor
   * @return the stream
   */
  public static Stream<Tensor> downStackTensors(Stream<Tensor> stream, int factor) {
    if (0 == factor) throw new IllegalArgumentException();
    if (-1 == factor) throw new IllegalArgumentException();
    return 1 == factor ? stream : stream.map(tensor -> {
      return GpuController.call(ctx -> {
        boolean expand = factor < 0;
        int abs = expand ? -factor : factor;
        return new ImgReshapeLayer(abs, abs, expand).eval(ctx, tensor);
      }).getData().get(0);
    });
  }
  
  /**
   * Down explode tensors stream.
   *
   * @param stream the stream
   * @param factor the factor
   * @return the stream
   */
  public static Stream<Tensor[]> downExplodeTensors(Stream<Tensor[]> stream, int factor) {
    if (0 >= factor) throw new IllegalArgumentException();
    if (-1 == factor) throw new IllegalArgumentException();
    return 1 == factor ? stream : stream.flatMap(tensor -> IntStream.range(0, factor * factor).mapToObj(subband -> {
      int[] select = new int[tensor[1].getDimensions()[2]];
      int offset = subband * select.length;
      for (int i = 0; i < select.length; i++) select[i] = offset + i;
      PipelineNetwork network = new PipelineNetwork();
      network.add(new ImgReshapeLayer(factor, factor, false));
      network.add(new ImgBandSelectLayer(select));
      Tensor result = GpuController.call(ctx ->
        network.eval(ctx, new Tensor[]{tensor[1]})).getData().get(0);
      return new Tensor[]{tensor[0], result};
    }));
  }
  
  /**
   * Build training model dag network.
   *
   * @param innerModel       the inner model
   * @param reproducedColumn the reproduced column
   * @param learnedColumn    the learned column
   * @return the dag network
   */
  public static DAGNetwork buildTrainingModel(NNLayer innerModel, int reproducedColumn, int learnedColumn) {
    PipelineNetwork network = new PipelineNetwork(Math.max(learnedColumn, reproducedColumn) + 1);
    // network.add(new NthPowerActivationLayer().setPower(0.5), );
    network.add(new MeanSqLossLayer(),
      network.add("image", innerModel, network.getInput(learnedColumn)),
      network.getInput(reproducedColumn));
    //addLogging(network);
    return network;
  }
  
  /**
   * Gets monitor.
   *
   * @param history the history
   * @return the monitor
   */
  public static TrainingMonitor getMonitor(List<Step> history) {
    return new TrainingMonitor() {
      @Override
      public void log(String msg) {
        System.out.println(msg); // Logged MnistProblemData
        TestUtil.rawOut.println(msg); // Realtime MnistProblemData
      }
      
      @Override
      public void onStepComplete(Step currentPoint) {
        history.add(currentPoint);
      }
      
      @Override
      public void clear() {
        super.clear();
      }
    };
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
            scalarStatistics.add(new Tensor(dimensions[0], dimensions[1]).fillByCoord(coord -> tensor.get(coord.coords[0], coord.coords[1], b)).getData());
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
        .filter(x -> x.coords[2] == band)
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
    tensor.coordStream().collect(Collectors.groupingBy(x -> x.coords[2], Collectors.toList()));
    Tensor normal = tensor.mapCoords((v, c) -> transform.apply(v, statistics[c.coords[2]]))
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
  
  /**
   * Get images tensor [ ] [ ].
   *
   * @param log        the log
   * @param size       the size
   * @param maxImages  the max images
   * @param categories the categories
   * @return the tensor [ ] [ ]
   */
  public static Tensor[][] getImages(NotebookOutput log, int size, int maxImages, String... categories) {
    log.out("Available images and categories:");
    log.code(() -> {
      return Caltech101.trainingDataStream().collect(Collectors.groupingBy(x -> x.label, Collectors.counting()));
    });
    int seed = (int) ((System.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    try {
      return Caltech101.trainingDataStream().filter(x -> {
        return Arrays.asList(categories).contains(x.label);
      }).map(labeledObj -> new Tensor[]{
        new Tensor(categories.length).set(Arrays.asList(categories).indexOf(labeledObj.label), 1.0),
        Tensor.fromRGB(resize(labeledObj.data.get(), size))
      }).sorted(Comparator.comparingInt(a -> System.identityHashCode(a) ^ seed)).limit(maxImages).toArray(i -> new Tensor[i][]);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * To 32.
   *
   * @param network the network
   */
  public static void to32(DAGNetwork network) {
    network.visitNodes(node -> {
      NNLayer layer = node.getLayer();
      if (layer instanceof ConvolutionLayer) {
        node.setLayer(layer.as(com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer.class));
      }
      else if (layer instanceof PoolingLayer) {
        node.setLayer(layer.as(com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer.class));
      }
    });
  }
  
  /**
   * To 64.
   *
   * @param network the network
   */
  public static void to64(DAGNetwork network) {
    network.visitNodes(node -> {
      NNLayer layer = node.getLayer();
      if (layer instanceof com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer) {
        node.setLayer(layer.as(ConvolutionLayer.class));
      }
      else if (layer instanceof com.simiacryptus.mindseye.layers.cudnn.f32.PoolingLayer) {
        node.setLayer(layer.as(PoolingLayer.class));
      }
    });
  }
  
  /**
   * Remove performance wrappers.
   *
   * @param log     the log
   * @param network the network
   */
  public static void removePerformanceWrappers(NotebookOutput log, DAGNetwork network) {
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
        return String.format("%s -> %.4f +- %.4f (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
      }).reduce((a, b) -> a + "\n\t" + b));
      System.out.println("Backward Performance: \n\t" + metrics.entrySet().stream().map(e -> {
        PercentileStatistics performance = e.getValue().getBackwardPerformance();
        return String.format("%s -> %.4f +- %.4f (%s)", e.getKey(), performance.getMean(), performance.getStdDev(), performance.getCount());
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
  public static void addPerformanceWrappers(NotebookOutput log, DAGNetwork network) {
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
}
