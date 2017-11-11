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
import com.simiacryptus.mindseye.layers.activation.*;
import com.simiacryptus.mindseye.layers.cudnn.CudaExecutionContext;
import com.simiacryptus.mindseye.layers.cudnn.f64.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.media.ImgReshapeLayer;
import com.simiacryptus.mindseye.layers.reducers.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.reducers.ProductInputsLayer;
import com.simiacryptus.mindseye.layers.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNetwork;
import com.simiacryptus.mindseye.network.graph.DAGNode;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.orient.OwlQn;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.text.TableOutput;
import com.simiacryptus.util.data.DoubleStatistics;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.junit.Test;
import org.junit.experimental.categories.Category;
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
 * The type Mnist test base.
 */
public class ImageEncodingTest {
  
  /**
   * Basic test.
   *
   * @throws Exception any exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void test() throws Exception {
    PrintStream originalOut = System.out;
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != originalOut) ((MarkdownNotebookOutput) log).addCopy(originalOut);
      
      int pretrainMinutes = 5;
      int timeoutMinutes = 10;
      int size = 256;
      
      Tensor[][] originalTrainingData = getImages(log, size, "kangaroo");
      log.h1("First Layer");
      InitializationStep step0 = new InitializationStep(log, originalTrainingData,
                                                                      50, size, pretrainMinutes, timeoutMinutes,
                                                                      3, 9, 7, 1).invoke();
  
      log.h1("Second Layer");
      AddLayerStep step1 = new AddLayerStep(log, step0.getTrainingData(), step0.getInnerModel(),
                                             2, step0.toSize, pretrainMinutes, timeoutMinutes,
                                             9, 13, 1, 2).invoke();
  
      log.h1("Third Layer");
      AddLayerStep step2 = new AddLayerStep(log, step1.getTrainingData(), step1.getIntegrationModel(),
                                             3, step1.toSize, pretrainMinutes, timeoutMinutes,
                                             13, 17, 7, 1).invoke();
  
    }
  }
  
  public static void initViaPCA(NotebookOutput log, TrainingMonitor monitor, Tensor[] tensorStream, ConvolutionLayer convolutionLayer, ImgBandBiasLayer biasLayer) {
    Tensor prototype = tensorStream[0];
    int[] dimensions = prototype.getDimensions();
    int[] filterDimensions = convolutionLayer.filter.getDimensions();
    assert filterDimensions[0] == dimensions[0];
    assert filterDimensions[1] == dimensions[1];
    int outputBands = dimensions[2];
    assert outputBands == biasLayer.getBias().length;
    int inputBands = filterDimensions[2] / outputBands;
    double[] averages = IntStream.range(0, outputBands).parallel().mapToDouble(b -> {
      return Arrays.stream(tensorStream).mapToDouble(tensor -> {
        return Arrays.stream(tensor.mapCoords((v, c) -> c.coords[2] == b ? v : Double.NaN).getData()).filter(Double::isFinite).average().getAsDouble();
      }).average().getAsDouble();
    }).toArray();
    biasLayer.setWeights(i -> {
      double v = averages[i];
      return Double.isFinite(v)?v:biasLayer.getBias()[i];
    });
    double[][] data = Arrays.stream(tensorStream).map(tensor -> {
      return tensor.mapCoords((v, c) -> v - averages[c.coords[2]]);
    }).map(x -> x.getData()).toArray(i -> new double[i][]);
    log.code(()->{
      RealMatrix realMatrix = MatrixUtils.createRealMatrix(data);
      Covariance covariance = new Covariance(realMatrix);
      RealMatrix covarianceMatrix = covariance.getCovarianceMatrix();
      EigenDecomposition decomposition = new EigenDecomposition(covarianceMatrix);
      int[] orderedVectors = IntStream.range(0, inputBands).mapToObj(x -> x)
                      .sorted(Comparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
      List<Tensor> rawComponents = IntStream.range(0, orderedVectors.length)
                                     .mapToObj(i -> {
                                       Tensor src = new Tensor(decomposition.getEigenvector(orderedVectors[i]).toArray(), dimensions[0], dimensions[1], outputBands).copy();
                                         return src
                                                  .scale(1.0/src.rms())
                                                  //.scale((decomposition.getRealEigenvalue(orderedVectors[inputBands-1])))
                                                  //.scale((decomposition.getRealEigenvalue(orderedVectors[i])))
                                                  //.scale((1.0 / decomposition.getRealEigenvalue(orderedVectors[0])))
                                                  .scale(Math.sqrt(6. / (inputBands + convolutionLayer.filter.dim() + 1)))
                                           ;
                                       }
                                     )
                                     .collect(Collectors.toList())
        ;
      Tensor[] vectors = rawComponents.stream().toArray(i->new Tensor[i]);
      convolutionLayer.filter.fillByCoord(c->{
        int outband = c.coords[2] / inputBands;
        int inband = c.coords[2] % inputBands;
        assert c.coords[0] < dimensions[0];
        assert c.coords[1] < dimensions[1];
        assert outband < outputBands;
        assert inband < inputBands;
        double v = vectors[inband].get(dimensions[0] - (c.coords[0]+1), dimensions[1]-(c.coords[1]+1), outputBands - (outband+1));
        return Double.isFinite(v)?v:convolutionLayer.filter.get(c);
      });
    });
  }
  
  public static Tensor[] convolutionFeatures(Stream<Tensor> tensors, int radius) {
    int padding = (radius - 1);
    return tensors.parallel().flatMap(image->{
      return IntStream.range(0, image.getDimensions()[0]-padding).filter(x->1==radius||0==x%(radius-1)).mapToObj(x->x).flatMap(x->{
        return IntStream.range(0, image.getDimensions()[1]-padding).filter(y->1==radius||0==y%(radius-1)).mapToObj(y->{
          Tensor region = new Tensor(radius, radius, image.getDimensions()[2]);
          final ToDoubleBiFunction<Double,Coordinate> f = (v, c)->{
            return image.get(c.coords[0] + x, c.coords[1] + y, c.coords[2]);
          };
          return region.mapCoords(f);
        });
      });
    }).toArray(i->new Tensor[i]);
  }
  
  public static Stream<Tensor> downshuffleTensors(Stream<Tensor> stream, int factor) {
    if(0 == factor) throw new IllegalArgumentException();
    if(-1 == factor) throw new IllegalArgumentException();
    return 1==factor?stream:stream.map(tensor->{
      return CudaExecutionContext.gpuContexts.run(ctx -> {
        boolean expand = factor < 0;
        int abs = expand?-factor:factor;
        return new ImgReshapeLayer(abs,abs,expand).eval(ctx, new Tensor[]{tensor});
      }).getData().get(0);
    });
  }
  
  public static DAGNetwork buildTrainingModel(NotebookOutput log, DAGNetwork innerModel, int reproducedColumn, int learnedColumn, int maskColumn, double factor_l1, double factor_entropy) {
    PipelineNetwork network = new PipelineNetwork(Math.max(maskColumn, Math.max(learnedColumn, reproducedColumn))+1);
    DAGNode input = 0 > maskColumn ? network.getInput(learnedColumn) : network.add(new ProductInputsLayer(), network.getInput(learnedColumn), network.getInput(maskColumn));
    DAGNode output = network.add("image", innerModel, input);
    DAGNode rmsError = network.add(new NthPowerActivationLayer().setPower(1.0 / 2.0),
      network.add(new MeanSqLossLayer(), output, network.getInput(reproducedColumn))
    );
    List<DAGNode> fitnessNodes = new ArrayList<>();
    fitnessNodes.add(rmsError);
    if(0<factor_entropy) {
      DAGNode density = network.add(new L1NormalizationLayer(),
        network.add(new SigmoidActivationLayer().setBalanced(true),
          network.add(new AbsActivationLayer(), input)));
      DAGNode entropy = network.add(new AbsActivationLayer(),
        network.add(new EntropyLossLayer(), density, density));
      fitnessNodes.add(network.add(new LinearActivationLayer().setScale(factor_entropy).freeze(), entropy));
    }
    if(0<factor_l1) {
      double lfactor = 1.0;
      DAGNode avgSignal = network.add(new NthPowerActivationLayer().setPower(1.0/lfactor),
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
  
  public static TrainingMonitor getMonitor(PrintStream originalOut, List<Step> history) {
    return new TrainingMonitor() {
      @Override
      public void log(String msg) {
        System.out.println(msg);
        if (null != originalOut && System.out != originalOut) originalOut.println(msg);
        super.log(msg);
      }
      
      @Override
      public void onStepComplete(Step currentPoint) {
        history.add(currentPoint);
        super.onStepComplete(currentPoint);
      }
      
      @Override
      public void clear() {
        super.clear();
      }
    };
  }
  
  public static void printDataStatistics(NotebookOutput log, Tensor[][] data) {
    for(int col=0;col<data[0].length;col++) {
      int c = col;
      log.out("Learned Representation Statistics for Column " + col + " (all bands)");
      log.code(()->{
        ScalarStatistics scalarStatistics = new ScalarStatistics();
        Arrays.stream(data)
          .flatMapToDouble(row-> Arrays.stream(row[c].getData()))
          .forEach(v->scalarStatistics.add(v));
        return scalarStatistics.getMetrics();
      });
      int _col = col;
      log.out("Learned Representation Statistics for Column " + col + " (by band)");
      log.code(()->{
        int[] dimensions = data[0][_col].getDimensions();
        return IntStream.range(0, dimensions[2]).mapToObj(x->x).flatMap(b->{
          return Arrays.stream(data).map(r->r[_col]).map(tensor->{
            ScalarStatistics scalarStatistics = new ScalarStatistics();
            scalarStatistics.add(new Tensor(dimensions[0], dimensions[1]).fillByCoord(coord -> tensor.get(coord.coords[0], coord.coords[1], b)).getData());
            return scalarStatistics;
          });
        }).map(x->x.getMetrics().toString()).reduce((a,b)->a+"\n"+b).get();
      });
    }
  }
  
  int modelNo = 0;
  private void printModel(NotebookOutput log, NNLayer network) {
    log.out("Learned Model Statistics: ");
    log.code(()->{
      ScalarStatistics scalarStatistics = new ScalarStatistics();
      network.state().stream().flatMapToDouble(x-> Arrays.stream(x))
        .forEach(v->scalarStatistics.add(v));
      return scalarStatistics.getMetrics();
    });
    String modelName = "model" + modelNo++ + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));
  }
  
  public static void printHistory(NotebookOutput log, List<Step> history) {
    if(!history.isEmpty()) {
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
  
  public List<NNLayer> dataPipeline = new ArrayList<>();
  public void validationReport(NotebookOutput log, Tensor[][] data) {
    log.out("Current dataset and evaluation results: ");
    log.code(() -> {
      TableOutput table = new TableOutput();
      Arrays.stream(data).map(tensorArray -> {
        LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
        for(int col=0;col<tensorArray.length;col++) {
          Tensor tensor = tensorArray[col];
          row.put("Data_"+col, render(log, tensor, 0<col));
          if(dataPipeline.size() >= col && 0 < col) {
            PipelineNetwork decoder = new PipelineNetwork();
            for(int i=col-1;i>=0;i--) {
              decoder.add(dataPipeline.get(i));
            }
            row.put("Decode_"+col, render(log, CudaExecutionContext.gpuContexts.run(ctx -> {
              return decoder.eval(ctx, new Tensor[]{tensor});
            }).getData().get(0), false));
          }
        }
        return row;
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
  }
  
  public static String render(NotebookOutput log, Tensor tensor, boolean normalize) {
    DoubleStatistics statistics = new DoubleStatistics();
    statistics.accept(tensor.getData());
    //Tensor normal = tensor.map(x -> 0xFF * (x - statistics.getMin()) / (statistics.getMax() - statistics.getMin()));
    Tensor normal = tensor.map(x -> 0x80 + 0x80 * (x - statistics.getAverage()) / (statistics.getStandardDeviation()))
                      .map(v->Math.min(0xFF,Math.max(0,v)));
    return (normalize?normal:tensor).toImages().stream().map(image -> {
      try {
        return log.image(image, "");
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }).reduce((a, b) -> a + b).get();
  }
  
  public static BufferedImage resize(BufferedImage source, int size) {
    BufferedImage image = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB);
    Graphics2D graphics = (Graphics2D) image.getGraphics();
    graphics.setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC));
    graphics.drawImage(source, 0, 0, size, size, null);
    return image;
  }
  
  public static Tensor[][] getImages(NotebookOutput log, int size, String... categories) {
    log.out("Available images and categories:");
    log.code(() -> {
      return Caltech101.trainingDataStream().collect(Collectors.groupingBy(x -> x.label, Collectors.counting()));
    });
    int seed = (int)((System.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    try {
      return Caltech101.trainingDataStream().filter(x -> {
        return Arrays.asList(categories).contains(x.label);
      }).map(labeledObj -> new Tensor[]{
        Tensor.fromRGB(resize(labeledObj.data.get(), size))
      }).sorted(Comparator.comparingInt(a -> System.identityHashCode(a) ^ seed)).toArray(i -> new Tensor[i][]);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  protected void train(NotebookOutput log, TrainingMonitor monitor, NNLayer network, Tensor[][] data, OrientationStrategy orientation, int timeoutMinutes, double factor_l1, boolean... mask) {
    log.out("Training for %s minutes, l1=%s, mask=%s", timeoutMinutes, factor_l1, Arrays.toString(mask));
    log.code(() -> {
      StochasticTrainable trainingSubject = new StochasticArrayTrainable(data, network, data.length);
      if(0 < factor_l1) trainingSubject = new ConstL12Normalizer(trainingSubject).setFactor_L1(factor_l1);
      trainingSubject = (StochasticTrainable) ((TrainableDataMask) trainingSubject).setMask(mask);
      new ValidatingTrainer(trainingSubject, new ArrayTrainable(data, network))
        .setMaxTrainingSize(data.length)
        .setMinTrainingSize(1)
        .setMonitor(monitor)
        .setOrientation(orientation)
        .setTimeout(timeoutMinutes, TimeUnit.MINUTES)
        .setLineSearchFactory(name->{
          if(name.contains("LBFGS") || name.contains("QQN")) {
            return new ArmijoWolfeSearch().setAlpha(1.0).setMaxAlpha(1e8);
          } else {
            return new ArmijoWolfeSearch().setMaxAlpha(1e6);
          }
        })
        .setMaxIterations(1000)
        .run();
    });
  }
  
  private class InitializationStep {
    private int fromSize;
    private int toSize;
    private NotebookOutput log;
    public final List<Step> history = new ArrayList<>();
  
    private TrainingMonitor monitor;
    private int pretrainMinutes;
    private int timeoutMinutes;
    private int images;
    private Tensor[][] originalTrainingData;
    private int band0;
    private int band1;
    private int radius;
    private int scale;
    private DAGNetwork innerModel;
    private Tensor[][] trainingData;
    
    public InitializationStep(NotebookOutput log, Tensor[][] originalTrainingData, int images, int fromSize, int pretrainMinutes, int timeoutMinutes, int band0, int band1, int radius, int scale) {
      this.log = log;
      this.monitor = getMonitor(System.out, history);
      this.monitor = monitor;
      this.pretrainMinutes = pretrainMinutes;
      this.timeoutMinutes = timeoutMinutes;
      this.band0 = band0;
      this.images = images;
      this.fromSize = fromSize;
      this.toSize = fromSize;
      this.originalTrainingData = originalTrainingData;
      this.band1 = band1;
      this.radius = radius;
      this.scale = scale;
    }
    
    public DAGNetwork getInnerModel() {
      return innerModel;
    }
    
    public Tensor[][] getTrainingData() {
      return trainingData;
    }
    
    public InitializationStep invoke() {
      ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(radius, radius, band1, band0 * scale * scale, false).setWeights(() -> 0.1 * (Math.random() - 0.5));
      ImgBandBiasLayer biasLayer1 = new ImgBandBiasLayer(band0 * scale * scale);
      innerModel = log.code(() -> {
        fromSize = toSize;
        toSize = (fromSize / scale + (radius - 1)); // 132
        PipelineNetwork network = new PipelineNetwork(1);
        network.add(convolutionLayer1);
        network.add(biasLayer1);
        int s = scale > 0 ? scale : -scale;
        if(s != 1) network.add(new ImgReshapeLayer(s, s, scale>0));
        network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        return network;
      });
      dataPipeline.add(innerModel);
      initViaPCA(log, monitor, convolutionFeatures(downshuffleTensors(Arrays.stream(originalTrainingData).map(x1 -> x1[0]), scale), radius), convolutionLayer1, biasLayer1);
      trainingData = Arrays.stream(originalTrainingData).limit(images).map(x -> new Tensor[]{
        x[0], new Tensor(toSize, toSize, band1).fill(() -> 0 * (Math.random() - 0.5))
      }).toArray(i -> new Tensor[i][]);
      
      {
        log.h2("Initialization");
        log.h3("Training");
        DAGNetwork trainingModel0 = buildTrainingModel(log, innerModel.copy().freeze(), 0, 1, -1, 0, 0);
        train(log, monitor, trainingModel0, trainingData, new QQN(), pretrainMinutes, 0, false, true);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData);
        printModel(log, innerModel);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      
      {
        log.h2("Tuning");
        log.h3("Training");
        DAGNetwork trainingModel0 = buildTrainingModel(log, innerModel, 0, 1, -1, 1e-5, 0);
        train(log, monitor, trainingModel0, trainingData, new OwlQn(), timeoutMinutes, 1e-12, false, true);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData);
        printModel(log, innerModel);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      
//      {
//        log.h2("Noising");
//        log.h3("Training");
//        double trainingDropout = 0;
//        double trainingNoise = 0.01;
//        int trainingExpansion = 5;
//        Tensor[][] noisyTrainingData = Arrays.stream(trainingData).flatMap(x -> {
//          return IntStream.range(0, trainingExpansion).mapToObj(i -> {
//            Tensor mask = x[1].map(v -> (Math.random() > trainingDropout ? 1 : 0) * Math.pow(2,trainingNoise * (Math.random() - 0.5)));
//            return new Tensor[]{
//              x[0], x[1], mask.scale(1.0/mask.sum())
//            };
//          });
//        }).toArray(i -> new Tensor[i][]);
//        DAGNetwork trainingModel0 = buildTrainingModel(log, innerModelA, 0, 1, 2, 1e-1, 1e-1);
//        train(log, monitor, trainingModel0, noisyTrainingData, new OwlQn(), timeoutMinutes, 1e-4, false, true, false);
//        printHistory(log, history);
//        log.h3("Results");
//        validationReport(log, trainingData, innerModelA);
//        printModel(log, innerModelA);
//        printDataStatistics(log, trainingData);
//        history.clear();
//      }
      
      return this;
    }
  
  }
  
  private class AddLayerStep {
    private final int fromSize;
    private final int toSize;
    private final ConvolutionLayer convolutionLayer;
    private final ImgBandBiasLayer biasLayer;
    private final PrintStream originalOut;
    private final NotebookOutput log;
    private final DAGNetwork priorModel;
    private final int layerNumber;
    private final int pretrainMinutes;
    private final int timeoutMinutes;
    private final int band1;
    private final int band2;
    private final int radius;
    private final int scale;
    private final List<Step> history;
    private final TrainingMonitor monitor;
    private Tensor[][] trainingData;
    private DAGNetwork innerModel;
    private PipelineNetwork integrationModel;
  
    public AddLayerStep(NotebookOutput log, Tensor[][] trainingData, DAGNetwork priorModel, int layerNumber, int fromSize, int pretrainMinutes, int timeoutMinutes, int band1, int band2, int radius, int scale) {
      this.originalOut = System.out;
      this.log = log;
      this.trainingData = trainingData;
      this.priorModel = priorModel;
      this.layerNumber = layerNumber;
      this.fromSize = fromSize;
      this.toSize = (fromSize / scale + (radius - 1)); // 70
      this.pretrainMinutes = pretrainMinutes;
      this.timeoutMinutes = timeoutMinutes;
      this.band1 = band1;
      this.band2 = band2;
      this.radius = radius;
      this.scale = scale;
      this.history = new ArrayList<>();
      this.monitor = ImageEncodingTest.getMonitor(originalOut, history);
      this.convolutionLayer = new ConvolutionLayer(radius, radius, band2, band1 * scale * scale, false).setWeights(()-> 0.01 * (Math.random() - 0.5));
      this.biasLayer = new ImgBandBiasLayer(band1 * scale * scale);
    }
    
    public Tensor[][] getTrainingData() {
      return trainingData;
    }
    
    public List<Step> getHistory() {
      return history;
    }
    
    public TrainingMonitor getMonitor() {
      return monitor;
    }
    
    public DAGNetwork getInnerModel() {
      return innerModel;
    }
    
    public AddLayerStep invoke() {
      innerModel = log.code(() -> {
        PipelineNetwork network = new PipelineNetwork(1);
        network.add(convolutionLayer);
        network.add(biasLayer);
        int s = scale > 0 ? scale : -scale;
        if(s != 1) network.add(new ImgReshapeLayer(s, s, scale>0));
        //network.add(new ActivationLayer(ActivationLayer.Mode.RELU));
        return network;
      });
      dataPipeline.add(innerModel);
      Tensor[] convolutionFeatures = convolutionFeatures(downshuffleTensors(Arrays.stream(trainingData).map(x -> x[layerNumber - 1]), scale), radius);
      initViaPCA(log, monitor, convolutionFeatures, convolutionLayer, biasLayer);
      this.integrationModel = log.code(() -> {
        PipelineNetwork network = new PipelineNetwork(1);
        network.add(innerModel);
        network.add(priorModel);
        return network;
      });
      trainingData = Arrays.stream(trainingData).map(x -> Stream.concat(
        Arrays.stream(x),
        Stream.of(new Tensor(toSize, toSize, band2).fill(() -> 0 * (Math.random() - 0.5))))
      .toArray(i->new Tensor[i])).toArray(i -> new Tensor[i][]);
      final boolean[] mask = new boolean[layerNumber+2];
      mask[layerNumber] = true;
      
      {
        log.h2("Initialization");
        log.h3("Training");
        DAGNetwork trainingModel0 = buildTrainingModel(log, innerModel.copy().freeze(), layerNumber-1, layerNumber, -1, 0, 0);
        train(log, monitor, trainingModel0, trainingData, new QQN(), pretrainMinutes, 0, mask);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData);
        printModel(log, innerModel);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      
      {
        log.h2("Tuning");
        log.h3("Training");
        DAGNetwork trainingModel0 = buildTrainingModel(log, innerModel, layerNumber-1, layerNumber, -1, 0, 0);
        train(log, monitor, trainingModel0, trainingData, new QQN(), timeoutMinutes, 0, mask);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData);
        printModel(log, innerModel);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      
      {
        log.h2("Integration Training");
        log.h3("Training");
        DAGNetwork trainingModel1 = buildTrainingModel(log, integrationModel, 0, layerNumber, -1, 0, 0);
        train(log, monitor, trainingModel1, trainingData, new QQN(), timeoutMinutes, 0, mask);
        printHistory(log, history);
        log.h3("Results");
        validationReport(log, trainingData);
        printModel(log, innerModel);
        printDataStatistics(log, trainingData);
        history.clear();
      }
      return this;
    }
  
    public PipelineNetwork getIntegrationModel() {
      return integrationModel;
    }
  }
}
