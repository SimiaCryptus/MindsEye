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

package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.java.ImgBandScaleLayer;
import com.simiacryptus.mindseye.layers.java.ImgBandSelectLayer;
import com.simiacryptus.mindseye.layers.java.ImgReshapeLayer;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.GifSequenceWriter;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image encoding util.
 */
public class EncodingUtil {
  private static final Logger log = LoggerFactory.getLogger(EncodingUtil.class);
  
  /**
   * The constant imageNumber.
   */
  public static int imageNumber = 0;
  /**
   * The constant svgNumber.
   */
  public static int svgNumber = 0;
  /**
   * The constant gifNumber.
   */
  public static int gifNumber = 0;
  /**
   * The constant out.
   */
  @javax.annotation.Nonnull
  protected static PrintStream rawOut = SysOutInterceptor.INSTANCE.getInner();
  
  /**
   * Add column tensor [ ] [ ].
   *
   * @param trainingData the training data
   * @param size         the size
   * @return the tensor [ ] [ ]
   */
  public static Tensor[][] addColumn(@javax.annotation.Nonnull final Tensor[][] trainingData, final int... size) {
    return Arrays.stream(trainingData).map(x -> Stream.concat(
      Arrays.stream(x),
      Stream.of(new Tensor(size).set(() -> 0.0 * (FastRandom.random() - 0.5))))
      .toArray(i -> new Tensor[i])).toArray(i -> new Tensor[i][]);
  }
  
  /**
   * Build training model dag network.
   *
   * @param innerModel       the heapCopy model
   * @param reproducedColumn the reproduced column
   * @param learnedColumn    the learned column
   * @return the dag network
   */
  @javax.annotation.Nonnull
  public static DAGNetwork buildTrainingModel(final NNLayer innerModel, final int reproducedColumn, final int learnedColumn) {
    @javax.annotation.Nonnull final PipelineNetwork network = new PipelineNetwork(Math.max(learnedColumn, reproducedColumn) + 1);
    // network.add(new NthPowerActivationLayer().setPower(0.5), );
    network.wrap(new MeanSqLossLayer(),
      network.add("image", innerModel, network.getInput(learnedColumn)),
      network.getInput(reproducedColumn));
    //addLogging(network);
    return network;
  }
  
  /**
   * Convolution features tensor [ ] [ ].
   *
   * @param tensors the tensors
   * @param radius  the radius
   * @param column  the column
   * @return the tensor [ ] [ ]
   */
  public static Stream<Tensor[]> convolutionFeatures(@javax.annotation.Nonnull final Stream<Tensor[]> tensors, final int radius, int column) {
    @javax.annotation.Nonnull final ThreadLocal<ConvolutionExtractor> extractors = ThreadLocal.withInitial(() -> new ConvolutionExtractor());
    return tensors.parallel().flatMap(row -> {
      @javax.annotation.Nonnull final Tensor region = new Tensor(radius, radius, row[column].getDimensions()[2]);
      return IntStream.range(0, row[column].getDimensions()[0] - (radius - 1)).mapToObj(x -> x).parallel().flatMap(x -> {
        return IntStream.range(0, row[column].getDimensions()[column] - (radius - 1)).mapToObj(y -> {
          final ConvolutionExtractor extractor = extractors.get();
          extractor.x = x;
          extractor.y = y;
          extractor.column = column;
          extractor.image = row;
          return new Tensor[]{row[0], region.mapCoords(extractor)};
        });
      });
    });
  }
  
  /**
   * Animated gif string.
   *
   * @param log              the log
   * @param baseline         the baseline
   * @param signedComponents the signed components
   * @return the string
   */
  public static String animatedGif(@javax.annotation.Nonnull final NotebookOutput log, @javax.annotation.Nonnull final Tensor baseline, @javax.annotation.Nonnull final List<Tensor> signedComponents) {
    int loopTimeMs = 15000;
    int framerate = 12;
    int frames = loopTimeMs * framerate / 1000;
    try {
      double step = 2 * Math.PI / frames;
      @javax.annotation.Nonnull String filename = EncodingUtil.gifNumber++ + ".gif";
      @javax.annotation.Nonnull File file = new File(log.getResourceDir(), filename);
      GifSequenceWriter.write(file, loopTimeMs / frames, true,
        DoubleStream.iterate(0, x -> x + step).limit(frames).parallel().mapToObj(t -> {
          return IntStream.range(0, signedComponents.size()).mapToObj(i -> {
            return signedComponents.get(i).scale((1 + Math.sin((1 + i) * t)) / 2);
          }).reduce((a, b) -> {
            Tensor add = a.addAndFree(b);
            b.freeRef();
            return add;
          }).get().add(baseline).toImage();
        }).toArray(i -> new BufferedImage[i]));
      return String.format("<img src=\"etc/%s\" />", filename);
    } catch (IOException e) {
      throw new RuntimeException(e);
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
  public static String decompositionSvg(@javax.annotation.Nonnull final NotebookOutput log, @javax.annotation.Nonnull final Tensor baseline, @javax.annotation.Nonnull final List<Tensor> signedComponents) {
    final List<DoubleStatistics> componentStats = signedComponents.stream().map(t -> new DoubleStatistics().accept(t.getData())).collect(Collectors.toList());
    @javax.annotation.Nonnull final String positiveFilter = IntStream.range(0, signedComponents.size()).mapToObj(i -> {
      String name;
      try {
        name = String.format("component_%s.png", EncodingUtil.imageNumber++);
        ImageIO.write(signedComponents.get(i).map(v -> v > 0 ? v * (0xFF / componentStats.get(i).getMax()) : 0).toImage(), "png", log.file(name));
      } catch (@javax.annotation.Nonnull final IOException e) {
        throw new RuntimeException(e);
      }
      return String.format("  <feImage xlink:href=\"%s\" result=\"pos_image_%s\" />\n", name, i);
    }).reduce((a, b) -> a + "\n" + b).get();
  
    @javax.annotation.Nonnull final String negativeFilter = IntStream.range(0, signedComponents.size()).mapToObj(i -> {
      String name;
      try {
        name = String.format("component_%s.png", EncodingUtil.imageNumber++);
        ImageIO.write(signedComponents.get(i).map(v -> v < 0 ? 0xFF - v * (0xFF / componentStats.get(i).getMin()) : 0).toImage(), "png", log.file(name));
      } catch (@javax.annotation.Nonnull final IOException e) {
        throw new RuntimeException(e);
      }
      return String.format("  <feImage xlink:href=\"%s\" result=\"neg_image_%s\" />\n", name, i);
    }).reduce((a, b) -> a + "\n" + b).get();
  
    @javax.annotation.Nonnull final String compositingFilters = IntStream.range(0, signedComponents.size()).mapToObj(i -> {
      final double fPos = componentStats.get(i).getMax() / 0xFF;
      final double fNeg = componentStats.get(i).getMin() / 0xFF;
      return "  <feComposite in=\"" + (i == 0 ? "FillPaint" : "lastResult") + "\" in2=\"neg_image_" + i + "\" result=\"lastResult\" operator=\"arithmetic\" k1=\"0.0\" k2=\"1.0\" k3=\"" + -fNeg + "\" k4=\"" + fNeg + "\"/>\n" +
        "  <feComposite in=\"lastResult\" in2=\"pos_image_" + i + "\" result=\"lastResult\" operator=\"arithmetic\" k1=\"0.0\" k2=\"1.0\" k3=\"" + fPos + "\" k4=\"" + 0.0 + "\"/>\n";
    }).reduce((a, b) -> a + "\n" + b).get();
  
    final int red = (int) baseline.get(0, 0, 0);
    final int green = (int) baseline.get(0, 0, 1);
    final int blue = (int) baseline.get(0, 0, 2);
    @javax.annotation.Nonnull final String avgHexColor = Long.toHexString(red + (green << 8) + (blue << 16));
    return "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n" +
      ("<defs>\n" +
        "<filter id=\"image\" >\n" + (
        positiveFilter + "\n" +
          negativeFilter + "\n" +
          compositingFilters
      ).replaceAll("\n", "\n\t") + "\n" +
        "</filter>\n" +
        "</defs>\n" +
        "<rect style=\"filter:url(#image);\" setByCoord=\"#" + avgHexColor + "\" width=\"256\" height=\"256\"/>"
      ).replaceAll("\n", "\n\t") +
      "\n</svg>";
  }
  
  /**
   * Down explode tensors stream.
   *
   * @param stream the stream
   * @param factor the factor
   * @return the stream
   */
  @javax.annotation.Nonnull
  public static Stream<Tensor[]> downExplodeTensors(@javax.annotation.Nonnull final Stream<Tensor[]> stream, final int factor) {
    if (0 >= factor) throw new IllegalArgumentException();
    if (-1 == factor) throw new IllegalArgumentException();
    return 1 == factor ? stream : stream.flatMap(tensor -> IntStream.range(0, factor * factor).mapToObj(subband -> {
      @javax.annotation.Nonnull final int[] select = new int[tensor[1].getDimensions()[2]];
      final int offset = subband * select.length;
      for (int i = 0; i < select.length; i++) {
        select[i] = offset + i;
      }
      @javax.annotation.Nonnull final PipelineNetwork network = new PipelineNetwork();
      network.add(new ImgReshapeLayer(factor, factor, false));
      network.add(new ImgBandSelectLayer(select));
      @javax.annotation.Nullable final Tensor result = network.eval(tensor[1]).getData().get(0);
      return new Tensor[]{tensor[0], result};
    }));
  }
  
  /**
   * Down stack tensors stream.
   *
   * @param stream the stream
   * @param factor the factor
   * @return the stream
   */
  @javax.annotation.Nonnull
  public static Stream<Tensor> downStackTensors(@javax.annotation.Nonnull final Stream<Tensor> stream, final int factor) {
    if (0 == factor) throw new IllegalArgumentException();
    if (-1 == factor) throw new IllegalArgumentException();
    return 1 == factor ? stream : stream.map(tensor -> {
      final boolean expand = factor < 0;
      final int abs = expand ? -factor : factor;
      return new ImgReshapeLayer(abs, abs, expand).eval(tensor).getData().get(0);
    });
  }
  
  /**
   * Find baseline tensor.
   *
   * @param decoder the decoder
   * @param tensor  the tensor
   * @return the tensor
   */
  @javax.annotation.Nullable
  public static Tensor findBaseline(@javax.annotation.Nonnull final PipelineNetwork decoder, @javax.annotation.Nonnull final Tensor tensor) {
    try {
      return decoder.eval(tensor.map(x -> 0)).getData().get(0);
    } catch (@javax.annotation.Nonnull final RuntimeException e) {
      throw e;
    } catch (@javax.annotation.Nonnull final Exception e) {
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
  @javax.annotation.Nullable
  public static Tensor findUnitComponent(final PipelineNetwork decoder, final int band, @javax.annotation.Nonnull final Tensor tensor) {
    @javax.annotation.Nonnull final PipelineNetwork decoderBand = new PipelineNetwork();
    @javax.annotation.Nonnull final double[] gate = new double[tensor.getDimensions()[2]];
    gate[band] = 1;
    decoderBand.add(new ImgBandScaleLayer(gate));
    decoderBand.add(decoder);
    try {
      return decoderBand.eval(tensor).getData().get(0);
      //return log.image(t.toImage(), "");
    } catch (@javax.annotation.Nonnull final RuntimeException e) {
      throw e;
    } catch (@javax.annotation.Nonnull final Exception e) {
      throw new RuntimeException(e);
    }
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
  public static Tensor[][] getImages(@javax.annotation.Nonnull final NotebookOutput log, final int size, final int maxImages, @javax.annotation.Nonnull final String... categories) {
    log.out("Available images and categories:");
    log.code(() -> {
      return Caltech101.trainingDataStream().collect(Collectors.groupingBy(x -> x.label, Collectors.counting()));
    });
    final int seed = (int) ((System.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    try {
      return Caltech101.trainingDataStream().filter(x -> {
        return categories.length == 0 || Arrays.asList(categories).contains(x.label);
      }).parallel().map(labeledObj -> {
        @Nullable BufferedImage img = labeledObj.data.get();
        img = TestUtil.resize(img, size);
        return new Tensor[]{
          0 == categories.length ? new Tensor() : new Tensor(categories.length).set(Arrays.asList(categories).indexOf(labeledObj.label), 1.0),
          Tensor.fromRGB(img)
        };
      }).sorted(Comparator.comparingInt(a -> System.identityHashCode(a) ^ seed)).limit(maxImages).toArray(i -> new Tensor[i][]);
    } catch (@javax.annotation.Nonnull final RuntimeException e) {
      throw e;
    } catch (@javax.annotation.Nonnull final IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Gets monitor.
   *
   * @param history the history
   * @return the monitor
   */
  public static TrainingMonitor getMonitor(@javax.annotation.Nonnull final List<StepRecord> history) {
    return new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }
      
      @Override
      public void log(final String msg) {
        log.info(msg); // Logged MnistProblemData
        EncodingUtil.rawOut.println(msg); // Realtime MnistProblemData
      }
      
      @Override
      public void onStepComplete(@javax.annotation.Nonnull final Step currentPoint) {
        history.add(new StepRecord(currentPoint.point.getMean(), currentPoint.time, currentPoint.iteration));
      }
    };
  }
  
  /**
   * Print model.
   *
   * @param log     the log
   * @param network the network
   * @param modelNo the model no
   */
  public static void printModel(@javax.annotation.Nonnull final NotebookOutput log, @javax.annotation.Nonnull final NNLayer network, final int modelNo) {
    log.out("Learned Model Statistics: ");
    log.code(() -> {
      @javax.annotation.Nonnull final ScalarStatistics scalarStatistics = new ScalarStatistics();
      network.state().stream().flatMapToDouble(x -> Arrays.stream(x))
        .forEach(v -> scalarStatistics.add(v));
      return scalarStatistics.getMetrics();
    });
    @javax.annotation.Nonnull final String modelName = "model" + modelNo + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));
  }
  
  /**
   * Render layer.
   *
   * @param log          the log
   * @param dataPipeline the data pipeline
   * @param row          the row
   * @param col          the col
   * @param tensor       the tensor
   */
  public static void renderLayer(@javax.annotation.Nonnull final NotebookOutput log, @javax.annotation.Nonnull final List<NNLayer> dataPipeline, @javax.annotation.Nonnull final LinkedHashMap<String, Object> row, final int col, @javax.annotation.Nonnull final Tensor tensor) {
    row.put("Data_" + col, com.simiacryptus.mindseye.test.TestUtil.render(log, tensor, 0 < col));
    if (dataPipeline.size() >= col - 1 && 1 < col) {
      @javax.annotation.Nonnull final PipelineNetwork decoder = new PipelineNetwork();
      for (int i = col - 2; i >= 0; i--) {
        decoder.add(dataPipeline.get(i));
      }
      @javax.annotation.Nullable final Tensor decoded = decoder.eval(tensor).getData().get(0);
      row.put("Decode_" + col, com.simiacryptus.mindseye.test.TestUtil.render(log, decoded, false));
      
      final List<Tensor> rawComponents = IntStream.range(0, tensor.getDimensions()[2])
        .mapToObj(band -> EncodingUtil.findUnitComponent(decoder, band, tensor))
        .collect(Collectors.toList());
      @javax.annotation.Nullable final Tensor baseline = EncodingUtil.findBaseline(decoder, tensor);
      final List<Tensor> signedComponents = IntStream.range(0, tensor.getDimensions()[2])
        .mapToObj(band -> rawComponents.get(band).minus(baseline))
        .collect(Collectors.toList());
      
      row.put("SVG_" + col, log.file(EncodingUtil.decompositionSvg(log, baseline, signedComponents), "svg" + EncodingUtil.svgNumber++ + ".svg", "SVG Composite Image"));
      row.put("GIF_" + col, EncodingUtil.animatedGif(log, baseline, signedComponents));
  
      @javax.annotation.Nonnull final String render = signedComponents.stream()
        .map(signedContribution -> com.simiacryptus.mindseye.test.TestUtil.render(log, signedContribution, true))
        .reduce((a, b) -> a + "" + b).get();
      row.put("Band_Decode_" + col, render);
    }
  }
  
  /**
   * Sets initial feature space.
   *
   * @param convolutionLayer the convolution layer
   * @param biasLayer        the bias layer
   * @param featureSpace     the feature space
   */
  public static void setInitialFeatureSpace(@javax.annotation.Nonnull final ConvolutionLayer convolutionLayer, @javax.annotation.Nonnull final ImgBandBiasLayer biasLayer, @javax.annotation.Nonnull final FindFeatureSpace featureSpace) {
    @javax.annotation.Nonnull final int[] filterDimensions = convolutionLayer.getKernel().getDimensions();
    final int outputBands = biasLayer.getBias().length;
    assert outputBands == biasLayer.getBias().length;
    final int inputBands = filterDimensions[2] / outputBands;
    biasLayer.setWeights(i -> {
      final double v = featureSpace.getAverages()[i];
      return Double.isFinite(v) ? v : biasLayer.getBias()[i];
    });
    final Tensor[] featureSpaceVectors = featureSpace.getVectors();
    for (@javax.annotation.Nonnull final Tensor t : featureSpaceVectors) {
      log.info(String.format("Feature Vector %s%n", t.prettyPrint()));
    }
    convolutionLayer.getKernel().setByCoord(c -> {
      final int kband = c.getCoords()[2];
      final int outband = kband % outputBands;
      final int inband = (kband - outband) / outputBands;
      assert outband < outputBands;
      assert inband < inputBands;
      int x = c.getCoords()[0];
      int y = c.getCoords()[1];
      x = filterDimensions[0] - (x + 1);
      y = filterDimensions[1] - (y + 1);
      final double v = featureSpaceVectors[inband].get(x, y, outband);
      return Double.isFinite(v) ? v : convolutionLayer.getKernel().get(c);
    });
    log.info(String.format("Bias: %s%n", Arrays.toString(biasLayer.getBias())));
    log.info(String.format("Kernel: %s%n", convolutionLayer.getKernel().prettyPrint()));
  }
  
  /**
   * Validation report.
   *
   * @param log          the log
   * @param data         the data
   * @param dataPipeline the data pipeline
   * @param maxRows      the max rows
   */
  public static void validationReport(@javax.annotation.Nonnull final NotebookOutput log, @javax.annotation.Nonnull final Tensor[][] data, @javax.annotation.Nonnull final List<NNLayer> dataPipeline, final int maxRows) {
    log.out("Current dataset and evaluation results: ");
    log.code(() -> {
      @javax.annotation.Nonnull final TableOutput table = new TableOutput();
      Arrays.stream(data).limit(maxRows).map(tensorArray -> {
        @javax.annotation.Nonnull final LinkedHashMap<String, Object> row = new LinkedHashMap<>();
        for (int col = 1; col < tensorArray.length; col++) {
          EncodingUtil.renderLayer(log, dataPipeline, row, col, tensorArray[col]);
        }
        return row;
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
  }
  
  private static class ConvolutionExtractor implements ToDoubleFunction<Coordinate> {
  
    /**
     * The Column.
     */
    public int column;
    /**
     * The Image.
     */
    public Tensor[] image;
    /**
     * The X.
     */
    public int x;
    /**
     * The Y.
     */
    public int y;
    
    @Override
    public double applyAsDouble(@javax.annotation.Nonnull final Coordinate c) {
      final int[] coords = c.getCoords();
      return image[column].get(coords[0] + x, coords[column] + y, coords[2]);
    }
  }
  
}
