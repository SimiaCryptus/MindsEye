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

package com.simiacryptus.mindseye.applications;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.GateBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ValueLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.models.CVPipe;
import com.simiacryptus.mindseye.models.CVPipe_VGG16;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.OrthonormalConstraint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.FileHTTPD;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.io.NullNotebookOutput;
import com.simiacryptus.util.lang.Tuple2;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * This notebook implements the Style Transfer protocol outlined in <a href="https://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a>
 *
 * @param <T> the type parameter
 * @param <U> the type parameter
 */
public abstract class ColorTransfer<T extends LayerEnum<T>, U extends CVPipe<T>> {
  
  private static final Logger logger = LoggerFactory.getLogger(ColorTransfer.class);
  /**
   * The Parallel loss functions.
   */
  public boolean parallelLossFunctions = true;
  private SimpleConvolutionLayer colorForwardTransform;
  private boolean ortho = true;
  private boolean unit = true;
  
  /**
   * Invert simple convolution layer.
   *
   * @param colorForwardTransform the color forward transform
   * @return the simple convolution layer
   */
  @Nonnull
  public static SimpleConvolutionLayer invert(final SimpleConvolutionLayer colorForwardTransform) {
    try {
      colorForwardTransform.assertAlive();
      SimpleConvolutionLayer invConv = new SimpleConvolutionLayer(1, 1, 9);
      RealMatrix matrix = getMatrix(colorForwardTransform.kernel);
      RealMatrix inverse = inverse(matrix);
      setMatrix(invConv.kernel, inverse);
      return invConv;
    } catch (Throwable e1) {
      logger.info("Error inverting kernel", e1);
      return unitTransformer();
    }
  }
  
  /**
   * Inverse real matrix.
   *
   * @param matrix the matrix
   * @return the real matrix
   */
  public static RealMatrix inverse(final RealMatrix matrix) {
    try {
      return MatrixUtils.inverse(matrix);
    } catch (Throwable e1) {
      logger.info("Error inverting kernel", e1);
      return new LUDecomposition(matrix).getSolver().getInverse();
    }
  }
  
  /**
   * Sets matrix.
   *
   * @param tensor the tensor
   * @param matrix the matrix
   */
  public static void setMatrix(final Tensor tensor, final RealMatrix matrix) {
    tensor.setByCoord(c -> {
      int b = c.getCoords()[2];
      int i = b % 3;
      int o = b / 3;
      return matrix.getEntry(i, o);
    });
  }
  
  /**
   * Gets matrix.
   *
   * @param tensor the tensor
   * @return the matrix
   */
  @Nonnull
  public static RealMatrix getMatrix(final Tensor tensor) {
    RealMatrix matrix = new BlockRealMatrix(3, 3);
    tensor.forEach((v, c) -> {
      int b = c.getCoords()[2];
      int i = b % 3;
      int o = b / 3;
      matrix.setEntry(i, o, v);
    }, false);
    return matrix;
  }
  
  /**
   * Unit transformer simple convolution layer.
   *
   * @return the simple convolution layer
   */
  @Nonnull
  public static SimpleConvolutionLayer unitTransformer() {
    return unitTransformer(3);
  }
  
  /**
   * Unit transformer simple convolution layer.
   *
   * @param bands the bands
   * @return the simple convolution layer
   */
  @Nonnull
  public static SimpleConvolutionLayer unitTransformer(final int bands) {
    SimpleConvolutionLayer colorForwardTransform = new SimpleConvolutionLayer(1, 1, bands * bands);
    colorForwardTransform.kernel.setByCoord(c -> {
      int band = c.getCoords()[2];
      int i = band % bands;
      int o = band / bands;
      return i == o ? 1.0 : 0.0;
    });
    return colorForwardTransform;
  }
  
  public static int[][] getIndexMap(final SimpleConvolutionLayer layer) {
    int[] kernelDimensions = layer.getKernelDimensions();
    double b = Math.sqrt(kernelDimensions[2]);
    int h = kernelDimensions[1];
    int w = kernelDimensions[0];
    int l = (int) (w * h * b);
    return IntStream.range(0, (int) b).mapToObj(i -> {
      return IntStream.range(0, l).map(j -> j + l * i).toArray();
    }).toArray(i -> new int[i][]);
  }
  
  /**
   * Style transfer buffered image.
   *
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param measureStyle    the measureStyle style
   * @return the buffered image
   */
  public Tensor transfer(
    final Tensor canvasImage,
    final StyleSetup<T> styleParameters,
    final int trainingMinutes,
    final NeuralSetup measureStyle
  )
  {
    return transfer(new NullNotebookOutput(), canvasImage, styleParameters, trainingMinutes, measureStyle, 50, true);
  }
  
  /**
   * Style transfer buffered image.
   *
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param measureStyle    the measureStyle style
   * @param maxIterations   the max iterations
   * @param verbose         the verbose
   * @return the buffered image
   */
  public Tensor transfer(
    @Nonnull final NotebookOutput log,
    final Tensor canvasImage,
    final StyleSetup<T> styleParameters,
    final int trainingMinutes,
    final NeuralSetup measureStyle,
    final int maxIterations,
    final boolean verbose
  )
  {
    canvasImage.assertAlive();
    try {
      //      log.p("Input Content:");
//      log.p(log.image(styleParameters.contentImage, "Content Image"));
//      log.p("Style Content:");
//      styleParameters.styleImages.forEach((file, styleImage) -> {
//        log.p(log.image(styleImage, file));
//      });
//      log.p("Input Canvas:");
//      log.p(log.image(canvasImage, "Input Canvas"));
      System.gc();
      TestUtil.monitorImage(canvasImage, false, false);
      String imageName = String.format("etc/image_%s.jpg", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
      log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", imageName, imageName));
      log.getHttpd().addHandler(imageName, "image/jpeg", r -> {
        try {
          ImageIO.write(canvasImage.toImage(), "jpeg", r);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      if (verbose) {
        log.p("Input Parameters:");
        log.code(() -> {
          return ArtistryUtil.toJson(styleParameters);
        });
      }
      this.setColorForwardTransform(train(log, styleParameters, trainingMinutes, measureStyle, maxIterations, verbose, canvasImage));
      try {
        ImageIO.write(canvasImage.toImage(), "jpeg", log.file(imageName));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      Tensor result = forwardTransform(canvasImage);
      log.p("Result:");
      log.p(log.image(result.toImage(), "Output Canvas"));
      return canvasImage.set(result);
    } catch (Throwable e) {
      logger.warn("Error in color transfer", e);
      return canvasImage;
    }
  }
  
  /**
   * Forward transform tensor.
   *
   * @param canvas the canvas
   * @return the tensor
   */
  @Nonnull
  public Tensor forwardTransform(final Tensor canvas) {
    Layer fwdTransform = getFwdTransform();
    if (null == fwdTransform) fwdTransform = unitTransformer();
    Tensor andFree = fwdTransform.eval(canvas).getDataAndFree().getAndFree(0);
    fwdTransform.freeRef();
    return andFree;
  }
  
  /**
   * Inverse transform tensor.
   *
   * @param canvas the canvas
   * @return the tensor
   */
  @Nonnull
  public Tensor inverseTransform(final Tensor canvas) {
    Layer invTransform = getInvTransform();
    Tensor andFree = invTransform.eval(canvas).getDataAndFree().getAndFree(0);
    invTransform.freeRef();
    return andFree;
  }
  
  /**
   * Gets fwd transform.
   *
   * @return the fwd transform
   */
  @Nonnull
  public Layer getFwdTransform() {
    SimpleConvolutionLayer colorForwardTransform = getColorForwardTransform();
    if (null == colorForwardTransform) return unitTransformer();
    return PipelineNetwork.wrap(
      1,
      colorForwardTransform,
      ArtistryUtil.getClamp(255)
    );
  }
  
  /**
   * Gets inv transform.
   *
   * @return the inv transform
   */
  @Nonnull
  public Layer getInvTransform() {
    PipelineNetwork network = new PipelineNetwork(1);
    SimpleConvolutionLayer colorForwardTransform = getColorForwardTransform();
    if (null == colorForwardTransform) return unitTransformer();
    network.wrap(
      ArtistryUtil.getClamp(255),
      network.wrap(
        invert(colorForwardTransform),
        network.getInput(0)
      )
    ).freeRef();
    return network;
  }
  
  /**
   * Train simple convolution layer.
   *
   * @param log             the log
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param measureStyle    the measureStyle style
   * @param maxIterations   the max iterations
   * @param verbose         the verbose
   * @param canvas          the canvas
   * @return the simple convolution layer
   */
  @Nonnull
  public SimpleConvolutionLayer train(
    @Nonnull final NotebookOutput log,
    final StyleSetup<T> styleParameters,
    final int trainingMinutes,
    final NeuralSetup measureStyle,
    final int maxIterations,
    final boolean verbose,
    final Tensor canvas
  )
  {
    NotebookOutput trainingLog = verbose ? log : new NullNotebookOutput();
    SimpleConvolutionLayer colorForwardTransform = unitTransformer();
    Trainable trainable = trainingLog.code(() -> {
      PipelineNetwork network = fitnessNetwork(measureStyle);
      network.setFrozen(true);
      TestUtil.instrumentPerformance(network);
      final FileHTTPD server = log.getHttpd();
      if (null != server) ArtistryUtil.addLayersHandler(network, server);
      PipelineNetwork trainingAssembly = new PipelineNetwork(1);
      trainingAssembly.wrap(
        network,
        trainingAssembly.wrap(
          ArtistryUtil.getClamp(255),
          trainingAssembly.add(
            colorForwardTransform,
            trainingAssembly.getInput(0)
          )
        )
      ).freeRef();
      ArtistryUtil.setPrecision(trainingAssembly, styleParameters.precision);
      Trainable trainable1 = new ArrayTrainable(trainingAssembly, 1).setVerbose(true).setMask(false).setData(Arrays.asList(new Tensor[][]{{canvas}}));
      trainingAssembly.freeRef();
      return trainable1;
    });
    try {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      String training_name = String.format("etc/training_%s.png", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
      log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", training_name, training_name));
      log.getHttpd().addHandler(training_name, "image/png", r -> {
        try {
          BufferedImage im = Util.toImage(TestUtil.plot(history));
          if (null != im) ImageIO.write(im, "png", r);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      trainingLog.code(() -> {
        new IterativeTrainer(trainable)
          .setMonitor(TestUtil.getMonitor(history))
          .setOrientation(getOrientation())
          .setMaxIterations(maxIterations)
          .setIterationsPerSample(100)
          .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1).setCurrentRate(1e0))
          .setTimeout(trainingMinutes, TimeUnit.MINUTES)
          .setTerminateThreshold(Double.NEGATIVE_INFINITY)
          .runAndFree();
        return TestUtil.plot(history);
      });
      try {
        BufferedImage image = Util.toImage(TestUtil.plot(history));
        if (null != image) ImageIO.write(image, "png", log.file(training_name));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    } finally {
      trainable.freeRef();
    }
    return colorForwardTransform;
  }
  
  @Nonnull
  public OrientationStrategy<LineSearchCursor> getOrientation() {
    return new TrustRegionStrategy(new LBFGS()) {
      @Override
      public TrustRegion getRegionPolicy(final Layer layer) {
        if (layer instanceof SimpleConvolutionLayer) {
          return new OrthonormalConstraint(getIndexMap((SimpleConvolutionLayer) layer)).setOrtho(isOrtho()).setUnit(isUnit());
        }
        return null;
      }
    };
  }
  
  /**
   * Gets style components.
   *
   * @param node          the node
   * @param network       the network
   * @param styleParams   the style params
   * @param mean          the mean
   * @param covariance    the covariance
   * @param centeringMode the centering mode
   * @return the style components
   */
  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(
    final DAGNode node,
    final PipelineNetwork network,
    final LayerStyleParams styleParams,
    final Tensor mean,
    final Tensor covariance,
    final CenteringMode centeringMode
  )
  {
    ArrayList<Tuple2<Double, DAGNode>> styleComponents = new ArrayList<>();
    if (null != styleParams && (styleParams.cov != 0 || styleParams.mean != 0 || styleParams.enhance != 0)) {
      InnerNode negTarget = null == mean ? null : network.wrap(new ValueLayer(mean.scale(-1)), new DAGNode[]{});
      node.addRef();
      InnerNode negAvg = network.wrap(new BandAvgReducerLayer().setAlpha(-1), node);
      if (styleParams.enhance != 0 || styleParams.cov != 0) {
        DAGNode recentered;
        switch (centeringMode) {
          case Origin:
            recentered = node;
            break;
          case Dynamic:
            negAvg.addRef();
            recentered = network.wrap(new GateBiasLayer(), node, negAvg);
            break;
          case Static:
            negTarget.addRef();
            recentered = network.wrap(new GateBiasLayer(), node, negTarget);
            break;
          default:
            throw new RuntimeException();
        }
        double covRms = null == covariance ? 1.0 : covariance.rms();
        if (styleParams.enhance != 0) {
          recentered.addRef();
          styleComponents.add(new Tuple2<>(
            -(0 == covRms ? styleParams.enhance : styleParams.enhance / covRms),
            network.wrap(
              new AvgReducerLayer(),
              network.wrap(
                new SquareActivationLayer(),
                recentered
              )
            )
          ));
        }
        if (styleParams.cov != 0) {
          int[] covDim = covariance.getDimensions();
          assert 0 < covDim[2] : Arrays.toString(covDim);
          int inputBands = mean.getDimensions()[2];
          assert 0 < inputBands : Arrays.toString(mean.getDimensions());
          int outputBands = covDim[2] / inputBands;
          assert 0 < outputBands : Arrays.toString(covDim) + " / " + inputBands;
          double covScale = 0 == covRms ? 1 : 1.0 / covRms;
          recentered.addRef();
          styleComponents.add(new Tuple2<>(styleParams.cov, network.wrap(
            new MeanSqLossLayer().setAlpha(covScale),
            network.wrap(new ValueLayer(covariance), new DAGNode[]{}),
            network.wrap(new GramianLayer(), recentered)
          )
          ));
        }
        recentered.freeRef();
      }
      else {
        node.freeRef();
      }
      if (styleParams.mean != 0) {
        double meanRms = mean.rms();
        double meanScale = 0 == meanRms ? 1 : 1.0 / meanRms;
        styleComponents.add(new Tuple2<>(
          styleParams.mean,
          network.wrap(new MeanSqLossLayer().setAlpha(meanScale), negAvg, negTarget)
        ));
      }
      else {
        if (null != negTarget) negTarget.freeRef();
        if (null != negAvg) negAvg.freeRef();
      }
    }
    else {
      node.freeRef();
    }
    return styleComponents;
  }
  
  /**
   * Measure style neural setup.
   *
   * @param style the style
   * @return the neural setup
   */
  public NeuralSetup measureStyle(final StyleSetup<T> style) {
    NeuralSetup self = new NeuralSetup(style);
    List<CharSequence> keyList = style.styleImages.keySet().stream().collect(Collectors.toList());
    Tensor contentInput = style.contentImage;
    List<Tensor> styleInputs = keyList.stream().map(x -> style.styleImages.get(x)).collect(Collectors.toList());
    IntStream.range(0, keyList.size()).forEach(i -> {
      self.styleTargets.put(keyList.get(i), new StyleTarget<>());
    });
    self.contentTarget = new ContentTarget<>();
    for (final T layerType : getLayerTypes()) {
      System.gc();
      Layer network = layerType.network();
      try {
        ArtistryUtil.setPrecision((DAGNetwork) network, style.precision);
        //network = new ImgTileSubnetLayer(network, 400,400,400,400);
        Tensor content = null == contentInput ? null : network.eval(contentInput).getDataAndFree().getAndFree(0);
        if (null != content) {
          self.contentTarget.content.put(layerType, content);
          logger.info(String.format("%s : target content = %s", layerType.name(), content.prettyPrint()));
          logger.info(String.format(
            "%s : content statistics = %s",
            layerType.name(),
            JsonUtil.toJson(new ScalarStatistics().add(content.getData()).getMetrics())
          ));
        }
        for (int i = 0; i < styleInputs.size(); i++) {
          Tensor styleInput = styleInputs.get(i);
          CharSequence key = keyList.get(i);
          StyleTarget<T> styleTarget = self.styleTargets.get(key);
          if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> x.getValue().params.get(
            layerType)).filter(x -> null != x).filter(x -> x.mean != 0 || x.cov != 0).count())
            continue;
          System.gc();
          Layer wrapAvg = ArtistryUtil.wrapTiledAvg(network.copy(), 400);
          Tensor mean = wrapAvg.eval(styleInput).getDataAndFree().getAndFree(0);
          wrapAvg.freeRef();
          styleTarget.mean.put(layerType, mean);
          logger.info(String.format("%s : style mean = %s", layerType.name(), mean.prettyPrint()));
          logger.info(String.format(
            "%s : mean statistics = %s",
            layerType.name(),
            JsonUtil.toJson(new ScalarStatistics().add(mean.getData()).getMetrics())
          ));
          if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> x.getValue().params.get(
            layerType)).filter(x -> null != x).filter(x -> x.cov != 0).count())
            continue;
          System.gc();
          Layer gram = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy()), 400);
          Tensor cov0 = gram.eval(styleInput).getDataAndFree().getAndFree(0);
          gram.freeRef();
          gram = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy(), mean), 400);
          Tensor cov1 = gram.eval(styleInput).getDataAndFree().getAndFree(0);
          gram.freeRef();
          styleTarget.cov0.put(layerType, cov0);
          styleTarget.cov1.put(layerType, cov1);
          int featureBands = mean.getDimensions()[2];
          int covarianceElements = cov1.getDimensions()[2];
          int selectedBands = covarianceElements / featureBands;
          logger.info(String.format(
            "%s : target cov0 = %s",
            layerType.name(),
            cov0.reshapeCast(featureBands, selectedBands, 1).prettyPrintAndFree()
          ));
          logger.info(String.format(
            "%s : cov0 statistics = %s",
            layerType.name(),
            JsonUtil.toJson(new ScalarStatistics().add(cov0.getData()).getMetrics())
          ));
          logger.info(String.format(
            "%s : target cov1 = %s",
            layerType.name(),
            cov1.reshapeCast(featureBands, selectedBands, 1).prettyPrintAndFree()
          ));
          logger.info(String.format(
            "%s : cov1 statistics = %s",
            layerType.name(),
            JsonUtil.toJson(new ScalarStatistics().add(cov1.getData()).getMetrics())
          ));
        }
      } finally {
        network.freeRef();
      }
    }
    styleInputs.forEach(ReferenceCountingBase::freeRef);
    if (null != contentInput) contentInput.freeRef();
    return self;
  }
  
  /**
   * Gets fitness components.
   *
   * @param setup   the setup
   * @param nodeMap the node buildMap
   * @return the fitness components
   */
  @Nonnull
  public List<Tuple2<Double, DAGNode>> getFitnessComponents(NeuralSetup setup, final Map<T, DAGNode> nodeMap) {
    List<Tuple2<Double, DAGNode>> functions = new ArrayList<>();
    functions.addAll(getContentComponents(setup, nodeMap));
    functions.addAll(getStyleComponents(setup, nodeMap));
    return functions;
  }
  
  /**
   * Gets style components.
   *
   * @param setup   the setup
   * @param nodeMap the node buildMap
   * @return the style components
   */
  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(NeuralSetup setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> styleComponents = new ArrayList<>();
    for (final List<CharSequence> keys : setup.style.styles.keySet()) {
      StyleTarget<T> styleTarget = keys.stream().map(x -> {
        StyleTarget<T> obj = setup.styleTargets.get(x);
        obj.addRef();
        return obj;
      }).reduce((a, b) -> {
        StyleTarget<T> r = a.add(b);
        a.freeRef();
        b.freeRef();
        return r;
      }).map(x -> {
        StyleTarget<T> r = x.scale(1.0 / keys.size());
        x.freeRef();
        return r;
      }).orElse(null);
      for (final T layerType : getLayerTypes()) {
        StyleCoefficients<T> styleCoefficients = setup.style.styles.get(keys);
        assert null != styleCoefficients;
        final DAGNode node = nodeMap.get(layerType);
        final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
        LayerStyleParams styleParams = styleCoefficients.params.get(layerType);
        Tensor mean = null == styleTarget ? null : styleTarget.mean.get(layerType);
        Tensor covariance;
        switch (styleCoefficients.centeringMode) {
          case Origin:
            covariance = null == styleTarget ? null : styleTarget.cov0.get(layerType);
            break;
          case Dynamic:
          case Static:
            covariance = null == styleTarget ? null : styleTarget.cov1.get(layerType);
            break;
          default:
            throw new RuntimeException();
        }
        node.addRef();
        styleComponents.addAll(getStyleComponents(node, network, styleParams, mean, covariance, styleCoefficients.centeringMode));
      }
      if (null != styleTarget) styleTarget.freeRef();
      
    }
    return styleComponents;
  }
  
  /**
   * Fitness function pipeline network.
   *
   * @param setup the setup
   * @return the pipeline network
   */
  @Nonnull
  public PipelineNetwork fitnessNetwork(NeuralSetup setup) {
    PipelineNetwork pipelineNetwork = getInstance().getNetwork();
    Map<T, DAGNode> nodes = new HashMap<>();
    Map<T, UUID> ids = getInstance().getNodes();
    ids.forEach((l, id) -> nodes.put(l, pipelineNetwork.getChildNode(id)));
    PipelineNetwork network = buildNetwork(setup, nodes, pipelineNetwork);
    ArtistryUtil.setPrecision(network, setup.style.precision);
    return network;
  }
  
  /**
   * Get layer types t [ ].
   *
   * @return the t [ ]
   */
  @Nonnull
  public abstract T[] getLayerTypes();
  
  /**
   * Gets content components.
   *
   * @param setup   the setup
   * @param nodeMap the node buildMap
   * @return the content components
   */
  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getContentComponents(NeuralSetup setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> contentComponents = new ArrayList<>();
    for (final T layerType : getLayerTypes()) {
      final DAGNode node = nodeMap.get(layerType);
      if (!setup.style.content.params.containsKey(layerType)) continue;
      final double coeff_content = setup.style.content.params.get(layerType);
      if (coeff_content != 0) {
        Tensor content = setup.contentTarget.content.get(layerType);
        if (content != null) {
          final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
          assert network != null;
          InnerNode innerNode = network.wrap(new MeanSqLossLayer().setAlpha(1.0 / content.rms()),
                                             node, network.wrap(new ValueLayer(content), new DAGNode[]{})
          );
          contentComponents.add(new Tuple2<>(coeff_content, innerNode));
        }
      }
    }
    return contentComponents;
  }
  
  /**
   * Gets instance.
   *
   * @return the instance
   */
  public abstract U getInstance();
  
  /**
   * Measure style pipeline network.
   *
   * @param setup   the setup
   * @param nodeMap the node buildMap
   * @param network the network
   * @return the pipeline network
   */
  public PipelineNetwork buildNetwork(NeuralSetup setup, final Map<T, DAGNode> nodeMap, final PipelineNetwork network) {
    List<Tuple2<Double, DAGNode>> functions = getFitnessComponents(setup, nodeMap);
    ArtistryUtil.reduce(network, functions, parallelLossFunctions);
    return network;
  }
  
  /**
   * Gets color forward transform.
   *
   * @return the color forward transform
   */
  public SimpleConvolutionLayer getColorForwardTransform() {
    SimpleConvolutionLayer colorForwardTransform = this.colorForwardTransform;
    if (null != colorForwardTransform) colorForwardTransform = (SimpleConvolutionLayer) colorForwardTransform.copy();
    return colorForwardTransform;
  }
  
  /**
   * Sets color forward transform.
   *
   * @param colorForwardTransform the color forward transform
   */
  public synchronized void setColorForwardTransform(SimpleConvolutionLayer colorForwardTransform) {
    colorForwardTransform.assertAlive();
    if (null != this.colorForwardTransform) this.colorForwardTransform.freeRef();
    this.colorForwardTransform = colorForwardTransform;
    if (null != this.colorForwardTransform) this.colorForwardTransform.addRef();
  }
  
  public boolean isOrtho() {
    return ortho;
  }
  
  public ColorTransfer<T, U> setOrtho(boolean ortho) {
    this.ortho = ortho;
    return this;
  }
  
  public boolean isUnit() {
    return unit;
  }
  
  public ColorTransfer<T, U> setUnit(boolean unit) {
    this.unit = unit;
    return this;
  }
  
  /**
   * The enum Centering mode.
   */
  public enum CenteringMode {
    /**
     * Dynamic centering mode.
     */
    Dynamic,
    /**
     * Static centering mode.
     */
    Static,
    /**
     * Origin centering mode.
     */
    Origin
  }
  
  /**
   * The type Vgg 16.
   */
  public static class VGG16 extends ColorTransfer<CVPipe_VGG16.Layer, CVPipe_VGG16> {
    
    public CVPipe_VGG16 getInstance() {
      return CVPipe_VGG16.INSTANCE;
    }
    
    @Nonnull
    public CVPipe_VGG16.Layer[] getLayerTypes() {
      return CVPipe_VGG16.Layer.values();
    }
    
  }
  
  /**
   * The type Vgg 19.
   */
  public static class VGG19 extends ColorTransfer<CVPipe_VGG19.Layer, CVPipe_VGG19> {
    
    public CVPipe_VGG19 getInstance() {
      return CVPipe_VGG19.INSTANCE;
    }
    
    @Nonnull
    public CVPipe_VGG19.Layer[] getLayerTypes() {
      return CVPipe_VGG19.Layer.values();
    }
    
  }
  
  /**
   * The type Content coefficients.
   *
   * @param <T> the type parameter
   */
  public static class ContentCoefficients<T extends LayerEnum<T>> {
    /**
     * The Params.
     */
    public final Map<T, Double> params = new HashMap<>();
    
    /**
     * Set content coefficients.
     *
     * @param l the l
     * @param v the v
     * @return the content coefficients
     */
    public ContentCoefficients set(final T l, final double v) {
      params.put(l, v);
      return this;
    }
    
  }
  
  /**
   * The type Layer style params.
   */
  public static class LayerStyleParams {
    /**
     * The Coeff style mean 0.
     */
    public final double mean;
    /**
     * The Coeff style bandCovariance 0.
     */
    public final double cov;
    private final double enhance;
    
    /**
     * Instantiates a new Layer style params.
     *
     * @param mean    the mean
     * @param cov     the bandCovariance
     * @param enhance the enhance
     */
    public LayerStyleParams(final double mean, final double cov, final double enhance) {
      this.mean = mean;
      this.cov = cov;
      this.enhance = enhance;
    }
  }
  
  /**
   * The type Style setup.
   *
   * @param <T> the type parameter
   */
  public static class StyleSetup<T extends LayerEnum<T>> {
    /**
     * The Precision.
     */
    public final Precision precision;
    /**
     * The Style image.
     */
    public final transient Map<CharSequence, Tensor> styleImages;
    /**
     * The Styles.
     */
    public final Map<List<CharSequence>, StyleCoefficients<T>> styles;
    /**
     * The Content.
     */
    public final ContentCoefficients<T> content;
    /**
     * The Content image.
     */
    public transient Tensor contentImage;
    
    
    /**
     * Instantiates a new Style setup.
     *
     * @param precision           the precision
     * @param contentImage        the content image
     * @param contentCoefficients the content coefficients
     * @param styleImages         the style image
     * @param styles              the styles
     */
    public StyleSetup(
      final Precision precision,
      final Tensor contentImage,
      ContentCoefficients<T> contentCoefficients,
      final Map<CharSequence, Tensor> styleImages,
      final Map<List<CharSequence>, StyleCoefficients<T>> styles
    )
    {
      this.precision = precision;
      this.contentImage = contentImage;
  
      this.styleImages = styleImages;
      if (!styleImages.values().stream().allMatch(x -> x instanceof Tensor)) throw new RuntimeException();
      this.styles = styles;
      this.content = contentCoefficients;
    }
    
  }
  
  /**
   * The type Style coefficients.
   *
   * @param <T> the type parameter
   */
  public static class StyleCoefficients<T extends LayerEnum<T>> {
    /**
     * The Dynamic center.
     */
    public final CenteringMode centeringMode;
    /**
     * The Params.
     */
    public final Map<T, LayerStyleParams> params = new HashMap<>();
    
    
    /**
     * Instantiates a new Style coefficients.
     *
     * @param centeringMode the dynamic center
     */
    public StyleCoefficients(final CenteringMode centeringMode) {
      this.centeringMode = centeringMode;
    }
    
    /**
     * Set style coefficients.
     *
     * @param layerType        the layer type
     * @param coeff_style_mean the coeff style mean
     * @param coeff_style_cov  the coeff style bandCovariance
     * @return the style coefficients
     */
    public StyleCoefficients set(final T layerType, final double coeff_style_mean, final double coeff_style_cov) {
      return set(
        layerType,
        coeff_style_mean,
        coeff_style_cov,
        0.0
      );
    }
    
    /**
     * Set style coefficients.
     *
     * @param layerType        the layer type
     * @param coeff_style_mean the coeff style mean
     * @param coeff_style_cov  the coeff style bandCovariance
     * @param dream            the dream
     * @return the style coefficients
     */
    public StyleCoefficients set(final T layerType, final double coeff_style_mean, final double coeff_style_cov, final double dream) {
      params.put(layerType, new LayerStyleParams(coeff_style_mean, coeff_style_cov, dream));
      return this;
    }
    
  }
  
  /**
   * The type Content target.
   *
   * @param <T> the type parameter
   */
  public static class ContentTarget<T extends LayerEnum<T>> {
    /**
     * The Content.
     */
    public Map<T, Tensor> content = new HashMap<>();
  }
  
  /**
   * The type Style target.
   *
   * @param <T> the type parameter
   */
  public class StyleTarget<T extends LayerEnum<T>> extends ReferenceCountingBase {
    /**
     * The Cov.
     */
    public Map<T, Tensor> cov0 = new HashMap<>();
    /**
     * The Cov.
     */
    public Map<T, Tensor> cov1 = new HashMap<>();
    /**
     * The Mean.
     */
    public Map<T, Tensor> mean = new HashMap<>();
    
    @Override
    protected void _free() {
      super._free();
      if (null != cov0) cov0.values().forEach(ReferenceCountingBase::freeRef);
      if (null != cov1) cov1.values().forEach(ReferenceCountingBase::freeRef);
      if (null != mean) mean.values().forEach(ReferenceCountingBase::freeRef);
    }
    
    /**
     * Add style target.
     *
     * @param right the right
     * @return the style target
     */
    public StyleTarget<T> add(StyleTarget<T> right) {
      StyleTarget<T> newStyle = new StyleTarget<>();
      Stream.concat(mean.keySet().stream(), right.mean.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = mean.get(layer);
        Tensor r = right.mean.get(layer);
        if (l != null && l != r) {
          Tensor add = l.add(r);
          newStyle.mean.put(layer, add);
        }
        else if (l != null) {
          l.addRef();
          newStyle.mean.put(layer, l);
        }
        else if (r != null) {
          r.addRef();
          newStyle.mean.put(layer, r);
        }
      });
      Stream.concat(cov0.keySet().stream(), right.cov0.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = cov0.get(layer);
        Tensor r = right.cov0.get(layer);
        if (l != null && l != r) {
          Tensor add = l.add(r);
          newStyle.cov0.put(layer, add);
        }
        else if (l != null) {
          l.addRef();
          newStyle.cov0.put(layer, l);
        }
        else if (r != null) {
          r.addRef();
          newStyle.cov0.put(layer, r);
        }
      });
      Stream.concat(cov1.keySet().stream(), right.cov1.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = cov1.get(layer);
        Tensor r = right.cov1.get(layer);
        if (l != null && l != r) {
          Tensor add = l.add(r);
          newStyle.cov1.put(layer, add);
        }
        else if (l != null) {
          l.addRef();
          newStyle.cov1.put(layer, l);
        }
        else if (r != null) {
          r.addRef();
          newStyle.cov1.put(layer, r);
        }
      });
      return newStyle;
    }
    
    /**
     * Scale style target.
     *
     * @param value the value
     * @return the style target
     */
    public StyleTarget<T> scale(double value) {
      StyleTarget<T> newStyle = new StyleTarget<>();
      mean.keySet().stream().distinct().forEach(layer -> {
        newStyle.mean.put(layer, mean.get(layer).scale(value));
      });
      cov0.keySet().stream().distinct().forEach(layer -> {
        newStyle.cov0.put(layer, cov0.get(layer).scale(value));
      });
      cov1.keySet().stream().distinct().forEach(layer -> {
        newStyle.cov1.put(layer, cov1.get(layer).scale(value));
      });
      return newStyle;
    }
    
  }
  
  /**
   * The type Neural setup.
   *
   */
  public class NeuralSetup {
    
    /**
     * The Style parameters.
     */
    public final StyleSetup<T> style;
    /**
     * The Content target.
     */
    public ContentTarget<T> contentTarget = new ContentTarget<>();
    /**
     * The Style targets.
     */
    public Map<CharSequence, StyleTarget<T>> styleTargets = new HashMap<>();
    
    
    /**
     * Instantiates a new Neural setup.
     *
     * @param style the style
     */
    public NeuralSetup(final StyleSetup<T> style) {this.style = style;}
  }
  
}
