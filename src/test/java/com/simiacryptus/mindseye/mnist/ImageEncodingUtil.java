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

package com.simiacryptus.mindseye.mnist;

import com.simiacryptus.mindseye.data.Caltech101;
import com.simiacryptus.mindseye.eval.*;
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.text.TableOutput;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.ToDoubleBiFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image encoding util.
 */
class ImageEncodingUtil {
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
  protected void printModel(NotebookOutput log, NNLayer network, final int modelNo) {
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
  protected void validationReport(NotebookOutput log, Tensor[][] data, List<NNLayer> dataPipeline, int maxRows) {
    log.out("Current dataset and evaluation results: ");
    log.code(() -> {
      TableOutput table = new TableOutput();
      Arrays.stream(data).limit(maxRows).map(tensorArray -> {
        LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
        for (int col = 1; col < tensorArray.length; col++) {
          Tensor tensor = tensorArray[col];
          row.put("Data_" + col, render(log, tensor, 0 < col));
          if (dataPipeline.size() >= col - 1 && 1 < col) {
            PipelineNetwork decoder = new PipelineNetwork();
            for (int i = col - 2; i >= 0; i--) {
              decoder.add(dataPipeline.get(i));
            }
            row.put("Decode_" + col, render(log, CudaExecutionContext.gpuContexts.run(ctx -> {
              return decoder.eval(ctx, tensor);
            }).getData().get(0), false));
            int bands = tensor.getDimensions()[2];
            String render = IntStream.range(0, bands).mapToObj(band -> {
              PipelineNetwork decoderBand = new PipelineNetwork();
              double[] gate = new double[bands];
              gate[band] = bands;
              decoderBand.add(new ImgBandScaleLayer(gate));
              decoderBand.add(decoder);
              try {
                Tensor t = CudaExecutionContext.gpuContexts.run(ctx -> {
                  return decoderBand.eval(ctx, tensor);
                }).getData().get(0);
                return render(log, t, true);
                //return log.image(t.toImage(), "");
              } catch (Exception e) {
                throw new RuntimeException(e);
              }
            }).reduce((a, b) -> a + "" + b).get();
            row.put("Band_Decode_" + col, render);
            
          }
        }
        return row;
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
  }
  
  /**
   * Train.
   *
   * @param log            the log
   * @param monitor        the monitor
   * @param network        the network
   * @param data           the data
   * @param orientation    the orientation
   * @param timeoutMinutes the timeout minutes
   * @param factor_l1      the factor l 1
   * @param mask           the mask
   */
  protected void train(NotebookOutput log, TrainingMonitor monitor, NNLayer network, Tensor[][] data, OrientationStrategy orientation, int timeoutMinutes, double factor_l1, boolean... mask) {
    log.out("Training for %s minutes, l1=%s, mask=%s", timeoutMinutes, factor_l1, Arrays.toString(mask));
    log.code(() -> {
      SampledTrainable trainingSubject = new SampledArrayTrainable(data, network, data.length);
      if (0 < factor_l1) trainingSubject = new ConstL12Normalizer(trainingSubject).setFactor_L1(factor_l1);
      trainingSubject = (SampledTrainable) ((TrainableDataMask) trainingSubject).setMask(mask);
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, new ArrayTrainable(data, network))
        .setMaxTrainingSize(data.length)
        .setMinTrainingSize(1)
        .setMonitor(monitor)
        .setTimeout(timeoutMinutes, TimeUnit.MINUTES)
        .setMaxIterations(1000);
      validatingTrainer.getRegimen().get(0)
        .setOrientation(orientation)
        .setLineSearchFactory(name -> {
          if (name.contains("LBFGS") || name.contains("QQN")) {
            return new ArmijoWolfeSearch().setAlpha(1.0).setMaxAlpha(1e8);
          }
          else {
            return new ArmijoWolfeSearch().setMaxAlpha(1e6);
          }
        });
      validatingTrainer
        .run();
    });
  }
  
  /**
   * Add column tensor [ ] [ ].
   *
   * @param trainingData the training data
   * @param size         the size
   * @return the tensor [ ] [ ]
   */
  protected Tensor[][] addColumn(Tensor[][] trainingData, int... size) {
    return Arrays.stream(trainingData).map(x -> Stream.concat(
      Arrays.stream(x),
      Stream.of(new Tensor(size).fill(() -> 0 * (FastRandom.random() - 0.5))))
      .toArray(i -> new Tensor[i])).toArray(i -> new Tensor[i][]);
  }
  
  /**
   * Initialize.
   *
   * @param log              the log
   * @param features         the features
   * @param convolutionLayer the convolution layer
   * @param biasLayer        the bias layer
   */
  protected void initialize(NotebookOutput log, Tensor[][] features, ConvolutionLayer convolutionLayer, ImgBandBiasLayer biasLayer) {
    Tensor prototype = features[0][1];
    int[] dimensions = prototype.getDimensions();
    int[] filterDimensions = convolutionLayer.filter.getDimensions();
    assert filterDimensions[0] == dimensions[0];
    assert filterDimensions[1] == dimensions[1];
    int outputBands = dimensions[2];
    assert outputBands == biasLayer.getBias().length;
    int inputBands = filterDimensions[2] / outputBands;
    FindFeatureSpace findFeatureSpace = findFeatureSpace(log, features, inputBands);
    setInitialFeatureSpace(convolutionLayer, biasLayer, findFeatureSpace);
  }
  
  /**
   * Find feature space find feature space.
   *
   * @param log        the log
   * @param features   the features
   * @param inputBands the input bands
   * @return the find feature space
   */
  protected FindFeatureSpace findFeatureSpace(NotebookOutput log, Tensor[][] features, int inputBands) {
    return new FindFeatureSpace(log, features, inputBands).invoke();
  }
  
  /**
   * Sets initial feature space.
   *
   * @param convolutionLayer the convolution layer
   * @param biasLayer        the bias layer
   * @param featureSpace     the feature space
   */
  protected void setInitialFeatureSpace(ConvolutionLayer convolutionLayer, ImgBandBiasLayer biasLayer, FindFeatureSpace featureSpace) {
    int[] filterDimensions = convolutionLayer.filter.getDimensions();
    int outputBands = biasLayer.getBias().length;
    assert outputBands == biasLayer.getBias().length;
    int inputBands = filterDimensions[2] / outputBands;
    biasLayer.setWeights(i -> {
      double v = featureSpace.getAverages()[i];
      return Double.isFinite(v) ? v : biasLayer.getBias()[i];
    });
    convolutionLayer.filter.fillByCoord(c -> {
      int outband = c.coords[2] / inputBands;
      int inband = c.coords[2] % inputBands;
      assert outband < outputBands;
      assert inband < inputBands;
      double v = featureSpace.getVectors()[inband].get(filterDimensions[0] - (c.coords[0] + 1), filterDimensions[1] - (c.coords[1] + 1), outputBands - (outband + 1));
      return Double.isFinite(v) ? v : convolutionLayer.filter.get(c);
    });
  }
  
  /**
   * Convolution features tensor [ ] [ ].
   *
   * @param tensors the tensors
   * @param radius  the radius
   * @return the tensor [ ] [ ]
   */
  protected Tensor[][] convolutionFeatures(Stream<Tensor[]> tensors, int radius) {
    return convolutionFeatures(tensors, radius, Math.max(3, radius));
  }
  
  protected Tensor[][] convolutionFeatures(Stream<Tensor[]> tensors, int radius, int padding) {
    return tensors.parallel().flatMap(image -> {
      return IntStream.range(0, image[1].getDimensions()[0] - (radius - 1)).filter(x -> 1 == radius || 0 == x % padding).mapToObj(x -> x).flatMap(x -> {
        return IntStream.range(0, image[1].getDimensions()[1] - (radius - 1)).filter(y -> 1 == radius || 0 == y % padding).mapToObj(y -> {
          Tensor region = new Tensor(radius, radius, image[1].getDimensions()[2]);
          final ToDoubleBiFunction<Double, Coordinate> f = (v, c) -> {
            return image[1].get(c.coords[0] + x, c.coords[1] + y, c.coords[2]);
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
  protected Stream<Tensor> downStackTensors(Stream<Tensor> stream, int factor) {
    if (0 == factor) throw new IllegalArgumentException();
    if (-1 == factor) throw new IllegalArgumentException();
    return 1 == factor ? stream : stream.map(tensor -> {
      return CudaExecutionContext.gpuContexts.run(ctx -> {
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
  protected Stream<Tensor[]> downExplodeTensors(Stream<Tensor[]> stream, int factor) {
    if (0 >= factor) throw new IllegalArgumentException();
    if (-1 == factor) throw new IllegalArgumentException();
    return 1 == factor ? stream : stream.flatMap(tensor -> IntStream.range(0, factor * factor).mapToObj(subband -> {
      int[] select = new int[tensor[1].getDimensions()[2]];
      int offset = subband * select.length;
      for (int i = 0; i < select.length; i++) select[i] = offset + i;
      PipelineNetwork network = new PipelineNetwork();
      network.add(new ImgReshapeLayer(factor, factor, false));
      network.add(new ImgBandSelectLayer(select));
      Tensor result = CudaExecutionContext.gpuContexts.run(ctx ->
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
   * @param factor_l1        the factor l 1
   * @param factor_entropy   the factor entropy
   * @return the dag network
   */
  protected DAGNetwork buildTrainingModel(NNLayer innerModel, int reproducedColumn, int learnedColumn, double factor_l1, double factor_entropy) {
    PipelineNetwork network = new PipelineNetwork(Math.max(learnedColumn, reproducedColumn) + 1);
    DAGNode input = network.getInput(learnedColumn);
    DAGNode output = network.add("image", innerModel, input);
    DAGNode rmsError = network.add(new NthPowerActivationLayer().setPower(1.0 / 2.0),
      network.add(new MeanSqLossLayer(), output, network.getInput(reproducedColumn))
    );
    List<DAGNode> fitnessNodes = new ArrayList<>();
    fitnessNodes.add(rmsError);
    if (0 < factor_entropy) {
      DAGNode density = network.add(new L1NormalizationLayer(),
        network.add(new SigmoidActivationLayer().setBalanced(true),
          network.add(new AbsActivationLayer(), input)));
      DAGNode entropy = network.add(new AbsActivationLayer(),
        network.add(new EntropyLossLayer(), density, density));
      fitnessNodes.add(network.add(new LinearActivationLayer().setScale(factor_entropy).freeze(), entropy));
    }
    if (0 < factor_l1) {
      double lfactor = 1.0;
      DAGNode avgSignal = network.add(new NthPowerActivationLayer().setPower(1.0 / lfactor),
        network.add(new AvgReducerLayer(),
          network.add(new NthPowerActivationLayer().setPower(lfactor),
            input)));
      fitnessNodes.add(network.add(new LinearActivationLayer().setScale(factor_l1).freeze(),
        network.add(new MeanSqLossLayer(),
          network.add(new NthPowerActivationLayer().setPower(1), avgSignal),
          network.add(new NthPowerActivationLayer().setPower(0), avgSignal)
        )));
    }
    network.add(new SumInputsLayer(), fitnessNodes.toArray(new DAGNode[]{}));
    return network;
  }
  
  /**
   * Gets monitor.
   *
   *
   * @param history     the history
   * @return the monitor
   */
  protected TrainingMonitor getMonitor(List<Step> history) {
    return new TrainingMonitor() {
      @Override
      public void log(String msg) {
        System.out.println(msg); // Logged Data
        ImageEncodingUtil.rawOut.println(msg); // Realtime Data
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
  protected void printDataStatistics(NotebookOutput log, Tensor[][] data) {
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
  protected void printHistory(NotebookOutput log, List<Step> history) {
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
  protected String render(NotebookOutput log, Tensor tensor, boolean normalize) {
    DoubleStatistics statistics = new DoubleStatistics();
    statistics.accept(tensor.getData());
    Tensor normal = tensor.map(x -> 0xFF * (x - statistics.getMin()) / (statistics.getMax() - statistics.getMin()))
      //Tensor normal = tensor.map(x -> 0x80 + 0x80 * (x - statistics.getAverage()) / (statistics.getStandardDeviation()))
      .map(v -> Math.min(0xFF, Math.max(0, v)));
    return (normalize ? normal : tensor).toImages().stream().map(image -> {
      try {
        return log.image(image, "");
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }).reduce((a, b) -> a + b).get();
  }
  
  /**
   * Resize buffered image.
   *
   * @param source the source
   * @param size   the size
   * @return the buffered image
   */
  protected BufferedImage resize(BufferedImage source, int size) {
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
  protected Tensor[][] getImages(NotebookOutput log, int size, int maxImages, String... categories) {
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
   * The type Find feature space.
   */
  protected class FindFeatureSpace {
    /**
     * The Log.
     */
    protected final NotebookOutput log;
    /**
     * The Features.
     */
    protected final Tensor[][] features;
    /**
     * The Input bands.
     */
    protected final int inputBands;
    /**
     * The Averages.
     */
    protected double[] averages;
    /**
     * The Vectors.
     */
    protected Tensor[] vectors;
  
    /**
     * Instantiates a new Find feature space.
     *
     * @param log        the log
     * @param features   the features
     * @param inputBands the input bands
     */
    public FindFeatureSpace(NotebookOutput log, Tensor[][] features, int inputBands) {
      this.log = log;
      this.features = features;
      this.inputBands = inputBands;
    }
  
    /**
     * Get averages double [ ].
     *
     * @return the double [ ]
     */
    public double[] getAverages() {
      return averages;
    }
  
    /**
     * Get vectors tensor [ ].
     *
     * @return the tensor [ ]
     */
    public Tensor[] getVectors() {
      return vectors;
    }
  
    /**
     * Invoke find feature space.
     *
     * @return the find feature space
     */
    public FindFeatureSpace invoke() {
      averages = findBandBias(features);
      Tensor[][] featureVectors = Arrays.stream(features).map(tensor -> {
        return new Tensor[]{tensor[0], tensor[1].mapCoords((v, c) -> v - averages[c.coords[2]])};
      }).toArray(i -> new Tensor[i][]);
      vectors = findFeatureSpace(log, featureVectors, inputBands);
      return this;
    }
  
    /**
     * Find band bias double [ ].
     *
     * @param features the features
     * @return the double [ ]
     */
    protected double[] findBandBias(Tensor[][] features) {
      Tensor prototype = features[0][1];
      int[] dimensions = prototype.getDimensions();
      int outputBands = dimensions[2];
      return IntStream.range(0, outputBands).parallel().mapToDouble(b -> {
        return Arrays.stream(features).mapToDouble(tensor -> {
          return Arrays.stream(tensor[1].mapCoords((v, c) -> c.coords[2] == b ? v : Double.NaN).getData()).filter(Double::isFinite).average().getAsDouble();
        }).average().getAsDouble();
      }).toArray();
    }
  
    /**
     * Find feature space tensor [ ].
     *
     * @param log            the log
     * @param featureVectors the feature vectors
     * @param components     the components
     * @return the tensor [ ]
     */
    protected Tensor[] findFeatureSpace(NotebookOutput log, Tensor[][] featureVectors, int components) {
      return log.code(() -> {
        int[] dimensions = featureVectors[0][1].getDimensions();
        double[][] data = Arrays.stream(featureVectors).map(x -> x[1].getData()).toArray(i -> new double[i][]);
        RealMatrix realMatrix = MatrixUtils.createRealMatrix(data);
        Covariance covariance = new Covariance(realMatrix);
        RealMatrix covarianceMatrix = covariance.getCovarianceMatrix();
        EigenDecomposition decomposition = new EigenDecomposition(covarianceMatrix);
        int[] orderedVectors = IntStream.range(0, components).mapToObj(x -> x)
          .sorted(Comparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
        return IntStream.range(0, orderedVectors.length)
          .mapToObj(i -> {
              Tensor src = new Tensor(decomposition.getEigenvector(orderedVectors[i]).toArray(), dimensions).copy();
              return src
                .scale(1.0 / src.rms())
                //.scale((decomposition.getRealEigenvalue(orderedVectors[inputBands-1])))
                //.scale((decomposition.getRealEigenvalue(orderedVectors[i])))
                //.scale((1.0 / decomposition.getRealEigenvalue(orderedVectors[0])))
                .scale(Math.sqrt(6. / (components + featureVectors[0][1].dim() + 1)))
                ;
            }
          ).toArray(i -> new Tensor[i]);
      });
    }
    
  }
}
