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
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.GpuController;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.SysOutInterceptor;

import javax.imageio.ImageIO;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image encoding util.
 */
class EncodingUtil {
  /**
   * The constant svgNumber.
   */
  public static int svgNumber = 0;
  /**
   * The constant imageNumber.
   */
  public static int imageNumber = 0;
  /**
   * The constant out.
   */
  protected static PrintStream rawOut = SysOutInterceptor.INSTANCE.getInner();
  
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
   * Render layer.
   *
   * @param log          the log
   * @param dataPipeline the data pipeline
   * @param row          the row
   * @param col          the col
   * @param tensor       the tensor
   */
  public static void renderLayer(NotebookOutput log, List<NNLayer> dataPipeline, LinkedHashMap<String, Object> row, int col, Tensor tensor) {
    row.put("Data_" + col, com.simiacryptus.mindseye.test.TestUtil.render(log, tensor, 0 < col));
    if (dataPipeline.size() >= col - 1 && 1 < col) {
      PipelineNetwork decoder = new PipelineNetwork();
      for (int i = col - 2; i >= 0; i--) {
        decoder.add(dataPipeline.get(i));
      }
      Tensor decoded = GpuController.call(ctx -> {
        return decoder.eval(ctx, tensor);
      }).getData().get(0);
      row.put("Decode_" + col, com.simiacryptus.mindseye.test.TestUtil.render(log, decoded, false));
      
      List<Tensor> rawComponents = IntStream.range(0, tensor.getDimensions()[2])
        .mapToObj(band -> findUnitComponent(decoder, band, tensor))
        .collect(Collectors.toList());
      Tensor baseline = findBaseline(decoder, tensor);
      List<Tensor> signedComponents = IntStream.range(0, tensor.getDimensions()[2])
        .mapToObj(band -> rawComponents.get(band).minus(baseline))
        .collect(Collectors.toList());
      
      row.put("SVG_" + col, log.file(decompositionSvg(log, baseline, signedComponents), "svg" + svgNumber++ + ".svg", "SVG Composite Image"));
      
      String render = signedComponents.stream()
        .map(signedContribution -> com.simiacryptus.mindseye.test.TestUtil.render(log, signedContribution, true))
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
  public static String decompositionSvg(NotebookOutput log, Tensor baseline, List<Tensor> signedComponents) {
    List<DoubleStatistics> componentStats = signedComponents.stream().map(t -> new DoubleStatistics().accept(t.getData())).collect(Collectors.toList());
    String positiveFilter = IntStream.range(0, signedComponents.size()).mapToObj(i -> {
      String name;
      try {
        name = String.format("component_%s.png", imageNumber++);
        ImageIO.write(signedComponents.get(i).map(v -> v > 0 ? (v * (0xFF / componentStats.get(i).getMax())) : 0).toImage(), "png", log.file(name));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      return String.format("  <feImage xlink:href=\"%s\" result=\"pos_image_%s\" />\n", name, i);
    }).reduce((a, b) -> a + "\n" + b).get();
    
    String negativeFilter = IntStream.range(0, signedComponents.size()).mapToObj(i -> {
      String name;
      try {
        name = String.format("component_%s.png", imageNumber++);
        ImageIO.write(signedComponents.get(i).map(v -> v < 0 ? (0xFF - (v * (0xFF / componentStats.get(i).getMin()))) : 0).toImage(), "png", log.file(name));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      return String.format("  <feImage xlink:href=\"%s\" result=\"neg_image_%s\" />\n", name, i);
    }).reduce((a, b) -> a + "\n" + b).get();
    
    String compositingFilters = IntStream.range(0, signedComponents.size()).mapToObj(i -> {
      double fPos = componentStats.get(i).getMax() / 0xFF;
      double fNeg = componentStats.get(i).getMin() / 0xFF;
      return "  <feComposite in=\"" + (i == 0 ? "FillPaint" : "lastResult") + "\" in2=\"neg_image_" + i + "\" result=\"lastResult\" operator=\"arithmetic\" k1=\"0.0\" k2=\"1.0\" k3=\"" + -fNeg + "\" k4=\"" + fNeg + "\"/>\n" +
        "  <feComposite in=\"lastResult\" in2=\"pos_image_" + i + "\" result=\"lastResult\" operator=\"arithmetic\" k1=\"0.0\" k2=\"1.0\" k3=\"" + fPos + "\" k4=\"" + 0.0 + "\"/>\n";
    }).reduce((a, b) -> a + "\n" + b).get();
    
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
      int kband = c.getCoords()[2];
      int outband = kband % outputBands;
      int inband = (kband - outband) / outputBands;
      assert outband < outputBands;
      assert inband < inputBands;
      int x = c.getCoords()[0];
      int y = c.getCoords()[1];
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
  public static Stream<Tensor[]> convolutionFeatures(Stream<Tensor[]> tensors, int radius) {
    int column = 1;
    ThreadLocal<ConvolutionExtractor> extractors = ThreadLocal.withInitial(() -> new ConvolutionExtractor());
    return tensors.parallel().flatMap(image -> {
      Tensor region = new Tensor(radius, radius, image[column].getDimensions()[2]);
      return IntStream.range(0, image[column].getDimensions()[0] - (radius - 1)).mapToObj(x -> x).parallel().flatMap(x -> {
        return IntStream.range(0, image[column].getDimensions()[column] - (radius - 1)).mapToObj(y -> {
          ConvolutionExtractor extractor = extractors.get();
          extractor.x = x;
          extractor.y = y;
          extractor.column = column;
          extractor.image = image;
          return new Tensor[]{image[0], region.mapCoords(extractor)};
        });
      });
    });
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
        EncodingUtil.rawOut.println(msg); // Realtime MnistProblemData
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
        Tensor.fromRGB(com.simiacryptus.mindseye.test.TestUtil.resize(labeledObj.data.get(), size))
      }).sorted(Comparator.comparingInt(a -> System.identityHashCode(a) ^ seed)).limit(maxImages).toArray(i -> new Tensor[i][]);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  private static class ConvolutionExtractor implements ToDoubleFunction<Coordinate> {
    
    /**
     * The X.
     */
    public int x;
    /**
     * The Y.
     */
    public int y;
    /**
     * The Column.
     */
    public int column;
    /**
     * The Image.
     */
    public Tensor[] image;
    
    @Override
    public double applyAsDouble(Coordinate c) {
      int[] coords = c.getCoords();
      return image[column].get(coords[0] + x, coords[column] + y, coords[2]);
    }
  }
  
}
