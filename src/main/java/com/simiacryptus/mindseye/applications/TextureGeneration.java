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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.GateBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.ValueLayer;
import com.simiacryptus.mindseye.models.CVPipe;
import com.simiacryptus.mindseye.models.CVPipe_VGG16;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.BisectionSearch;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.RangeConstraint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.StreamNanoHTTPD;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.io.NullNotebookOutput;
import com.simiacryptus.util.lang.Tuple2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * This notebook implements the Style Transfer protocol outlined in <a href="https://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a>
 *
 * @param <T> the type parameter
 * @param <U> the type parameter
 */
public abstract class TextureGeneration<T extends LayerEnum<T>, U extends CVPipe<T>> {
  
  private static final Logger logger = LoggerFactory.getLogger(TextureGeneration.class);
  /**
   * The Parallel loss functions.
   */
  public boolean parallelLossFunctions = true;
  private boolean tiled;
  
  public TextureGeneration() {tiled = true;}
  
  public static BufferedImage generate(@Nonnull final NotebookOutput log, final VGG19 styleTransfer, final Precision precision, final AtomicInteger imageSize, final double growthFactor, final Map<List<CharSequence>, StyleCoefficients> styles, final int trainingMinutes, BufferedImage canvasImage, final int phases, final int maxIterations, final StreamNanoHTTPD server) {
    log.h1("Phase 0");
    Map<CharSequence, BufferedImage> styleImages = new HashMap<>();
    StyleSetup styleSetup;
    NeuralSetup measureStyle;
    
    styleImages.clear();
    styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(x -> x, image -> ArtistryUtil.load(image, imageSize.get()))));
    styleSetup = new StyleSetup(precision, styleImages, styles);
    measureStyle = styleTransfer.measureStyle(styleSetup);
    
    canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
    canvasImage = styleTransfer.generate(server, log, canvasImage, styleSetup, trainingMinutes, measureStyle, maxIterations);
    for (int i = 1; i < phases; i++) {
      log.h1("Phase " + i);
      imageSize.set((int) (imageSize.get() * growthFactor));
      
      styleImages.clear();
      styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(x -> x, image -> ArtistryUtil.load(image, imageSize.get()))));
      styleSetup = new StyleSetup(precision, styleImages, styles);
      measureStyle = styleTransfer.measureStyle(styleSetup);
      
      canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
      canvasImage = styleTransfer.generate(server, log, canvasImage, styleSetup, trainingMinutes, measureStyle, maxIterations);
    }
    return canvasImage;
  }
  
  @Nonnull
  public static BufferedImage initCanvas(final AtomicInteger imageSize) {
    return ArtistryUtil.paint_Plasma(imageSize.get(), 3, 100.0, 1.4).toImage();
  }
  
  /**
   * Style transfer buffered image.
   *
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param measureStyle    the measure style
   * @return the buffered image
   */
  public BufferedImage generate(final BufferedImage canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes, final NeuralSetup measureStyle) {
    return generate(null, new NullNotebookOutput(), canvasImage, styleParameters, trainingMinutes, measureStyle, 50);
  }
  
  /**
   * Style transfer buffered image.
   *
   * @param server          the server
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param measureStyle    the measure style
   * @param maxIterations
   * @return the buffered image
   */
  public BufferedImage generate(final StreamNanoHTTPD server, @Nonnull final NotebookOutput log, final BufferedImage canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes, final NeuralSetup measureStyle, final int maxIterations) {
    BufferedImage result = ArtistryUtil.logExceptionWithDefault(log, () -> {
      System.gc();
      Tensor canvas = Tensor.fromRGB(canvasImage);
      TestUtil.monitorImage(canvas, false, false);
      log.p("Input Parameters:");
      log.code(() -> {
        return ArtistryUtil.toJson(styleParameters);
      });
      PipelineNetwork network = fitnessNetwork(measureStyle);
      network.setFrozen(true);
      ArtistryUtil.setPrecision(network, styleParameters.precision);
      TestUtil.instrumentPerformance(network);
      if (null != server) ArtistryUtil.addLayersHandler(network, server);
      if (tiled) network = ArtistryUtil.tileCycle(network);
      train(log, canvas, network, trainingMinutes, maxIterations);
      return canvas.toImage();
    }, canvasImage);
    log.p("Result:");
    log.p(log.image(result, "Result"));
    return result;
  }
  
  public void train(@Nonnull final NotebookOutput log, final Tensor canvas, final PipelineNetwork network, final int trainingMinutes, final int maxIterations) {
    Trainable trainable = new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
    log.code(() -> {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      new IterativeTrainer(trainable)
        .setMonitor(TestUtil.getMonitor(history))
        //        .setOrientation(new QQN())
        .setOrientation(new TrustRegionStrategy() {
          @Override
          public TrustRegion getRegionPolicy(final com.simiacryptus.mindseye.lang.Layer layer) {
            return new RangeConstraint().setMin(1e-2).setMax(256);
          }
        })
        .setMaxIterations(maxIterations)
        .setIterationsPerSample(100)
        //        .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
        .setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e6))
        //        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
        .setTimeout(trainingMinutes, TimeUnit.MINUTES)
        .setTerminateThreshold(Double.NEGATIVE_INFINITY)
        .runAndFree();
      return TestUtil.plot(history);
    });
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
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(final DAGNode node, final PipelineNetwork network, final LayerStyleParams styleParams, final Tensor mean, final Tensor covariance, final CenteringMode centeringMode) {
    ArrayList<Tuple2<Double, DAGNode>> styleComponents = new ArrayList<>();
    if (null != styleParams && (styleParams.cov != 0 || styleParams.mean != 0)) {
      double meanRms = mean.rms();
      double meanScale = 0 == meanRms ? 1 : (1.0 / meanRms);
      InnerNode negTarget = network.wrap(new ValueLayer(mean.scale(-1)), new DAGNode[]{});
      InnerNode negAvg = network.wrap(new BandAvgReducerLayer().setAlpha(-1), node);
      if (styleParams.cov != 0) {
        DAGNode recentered;
        switch (centeringMode) {
          case Origin:
            recentered = node;
            break;
          case Dynamic:
            recentered = network.wrap(new GateBiasLayer(), node, negAvg);
            break;
          case Static:
            recentered = network.wrap(new GateBiasLayer(), node, negTarget);
            break;
          default:
            throw new RuntimeException();
        }
        int[] covDim = covariance.getDimensions();
        assert 0 < covDim[2] : Arrays.toString(covDim);
        int inputBands = mean.getDimensions()[2];
        assert 0 < inputBands : Arrays.toString(mean.getDimensions());
        int outputBands = covDim[2] / inputBands;
        assert 0 < outputBands : Arrays.toString(covDim) + " / " + inputBands;
        double covRms = covariance.rms();
        double covScale = 0 == covRms ? 1 : (1.0 / covRms);
        styleComponents.add(new Tuple2<>(styleParams.cov, network.wrap(new MeanSqLossLayer().setAlpha(covScale),
          network.wrap(new ValueLayer(covariance), new DAGNode[]{}),
          network.wrap(new GramianLayer(), recentered))
        ));
      }
      if (styleParams.mean != 0) {
        styleComponents.add(new Tuple2<>(styleParams.mean,
          network.wrap(new MeanSqLossLayer().setAlpha(meanScale), negAvg, negTarget)
        ));
      }
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
    NeuralSetup<T> self = new NeuralSetup(style);
    List<CharSequence> keyList = style.styleImages.keySet().stream().collect(Collectors.toList());
    List<Tensor> styleInputs = keyList.stream().map(x -> style.styleImages.get(x)).map(img -> Tensor.fromRGB(img)).collect(Collectors.toList());
    IntStream.range(0, keyList.size()).forEach(i -> {
      self.styleTargets.put(keyList.get(i), new StyleTarget());
    });
    self.contentTarget = new ContentTarget();
    for (final T layerType : getLayerTypes()) {
      System.gc();
      final PipelineNetwork network = layerType.texture();
      ArtistryUtil.setPrecision(network, style.precision);
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        CharSequence key = keyList.get(i);
        StyleTarget<T> styleTarget = self.styleTargets.get(key);
        if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> (LayerStyleParams) x.getValue().params.get(layerType)).filter(x -> null != x).filter(x -> x.mean != 0 || x.cov != 0).count())
          continue;
        System.gc();
        Tensor mean = ArtistryUtil.wrapAvg(network.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        styleTarget.mean.put(layerType, mean);
        logger.info(String.format("%s : style mean = %s", layerType.name(), mean.prettyPrint()));
        logger.info(String.format("%s : mean statistics = %s", layerType.name(), JsonUtil.toJson(new ScalarStatistics().add(mean.getData()).getMetrics())));
        if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> (LayerStyleParams) x.getValue().params.get(layerType)).filter(x -> null != x).filter(x -> x.cov != 0).count())
          continue;
        System.gc();
        Tensor cov0 = ArtistryUtil.gram(network.copy()).eval(styleInput).getDataAndFree().getAndFree(0);
        Tensor cov1 = ArtistryUtil.gram(network.copy(), mean).eval(styleInput).getDataAndFree().getAndFree(0);
        styleTarget.cov0.put(layerType, cov0);
        styleTarget.cov1.put(layerType, cov1);
        int featureBands = mean.getDimensions()[2];
        int covarianceElements = cov1.getDimensions()[2];
        int selectedBands = covarianceElements / featureBands;
        logger.info(String.format("%s : target cov0 = %s", layerType.name(), cov0.reshapeCast(featureBands, selectedBands, 1).prettyPrint()));
        logger.info(String.format("%s : cov0 statistics = %s", layerType.name(), JsonUtil.toJson(new ScalarStatistics().add(cov0.getData()).getMetrics())));
        logger.info(String.format("%s : target cov1 = %s", layerType.name(), cov1.reshapeCast(featureBands, selectedBands, 1).prettyPrint()));
        logger.info(String.format("%s : cov1 statistics = %s", layerType.name(), JsonUtil.toJson(new ScalarStatistics().add(cov1.getData()).getMetrics())));
      }
    }
    return self;
  }
  
  /**
   * Gets fitness components.
   *
   * @param setup   the setup
   * @param nodeMap the node map
   * @return the fitness components
   */
  @Nonnull
  public List<Tuple2<Double, DAGNode>> getFitnessComponents(NeuralSetup setup, final Map<T, DAGNode> nodeMap) {
    List<Tuple2<Double, DAGNode>> functions = new ArrayList<>();
    functions.addAll(new ArrayList<>());
    functions.addAll(getStyleComponents(setup, nodeMap));
    return functions;
  }
  
  /**
   * Gets style components.
   *
   * @param setup   the setup
   * @param nodeMap the node map
   * @return the style components
   */
  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> styleComponents = new ArrayList<>();
    for (final T layerType : getLayerTypes())
      for (final List<CharSequence> keys : setup.style.styles.keySet()) {
        StyleTarget<T> styleTarget = keys.stream().map(x -> setup.styleTargets.get(x)).reduce((a, b) -> a.add(b)).map(x -> x.scale(1.0 / keys.size())).get();
        StyleCoefficients<T> styleCoefficients = setup.style.styles.get(keys);
        assert null != styleCoefficients;
        assert null != styleTarget;
        final DAGNode node = nodeMap.get(layerType);
        final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
        LayerStyleParams styleParams = styleCoefficients.params.get(layerType);
        Tensor mean = styleTarget.mean.get(layerType);
        
        Tensor covariance;
        switch (styleCoefficients.centeringMode) {
          case Origin:
            covariance = styleTarget.cov0.get(layerType);
            break;
          case Dynamic:
          case Static:
            covariance = styleTarget.cov1.get(layerType);
            break;
          default:
            throw new RuntimeException();
        }
        styleComponents.addAll(getStyleComponents(node, network, styleParams, mean, covariance, styleCoefficients.centeringMode));
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
    //network = withClamp(network);
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
   * Gets instance.
   *
   * @return the instance
   */
  public abstract U getInstance();
  
  /**
   * Measure style pipeline network.
   *
   * @param setup   the setup
   * @param nodeMap the node map
   * @param network the network
   * @return the pipeline network
   */
  public PipelineNetwork buildNetwork(NeuralSetup setup, final Map<T, DAGNode> nodeMap, final PipelineNetwork network) {
    List<Tuple2<Double, DAGNode>> functions = getFitnessComponents(setup, nodeMap);
    ArtistryUtil.reduce(network, functions, parallelLossFunctions);
    return network;
  }
  
  public boolean isTiled() {
    return tiled;
  }
  
  public TextureGeneration<T, U> setTiled(boolean tiled) {
    this.tiled = tiled;
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
  public static class VGG16 extends TextureGeneration<CVPipe_VGG16.Layer, CVPipe_VGG16> {
    
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
  public static class VGG19 extends TextureGeneration<CVPipe_VGG19.Layer, CVPipe_VGG19> {
    
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
     * The Coeff style cov 0.
     */
    public final double cov;
    
    /**
     * Instantiates a new Layer style params.
     *
     * @param mean the mean
     * @param cov  the cov
     */
    public LayerStyleParams(final double mean, final double cov) {
      this.mean = mean;
      this.cov = cov;
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
    public final transient Map<CharSequence, BufferedImage> styleImages;
    /**
     * The Styles.
     */
    public final Map<List<CharSequence>, StyleCoefficients> styles;
    
    
    /**
     * Instantiates a new Style setup.
     *
     * @param precision   the precision
     * @param styleImages the style image
     * @param styles      the styles
     */
    public StyleSetup(final Precision precision, final Map<CharSequence, BufferedImage> styleImages, final Map<List<CharSequence>, StyleCoefficients> styles) {
      this.precision = precision;
      this.styleImages = styleImages;
      this.styles = styles;
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
     * @param coeff_style_cov  the coeff style cov
     * @return the style coefficients
     */
    public StyleCoefficients set(final T layerType, final double coeff_style_mean, final double coeff_style_cov) {
      params.put(layerType, new LayerStyleParams(coeff_style_mean, coeff_style_cov));
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
  public static class StyleTarget<T extends LayerEnum<T>> {
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
        if (l != null && l != r) newStyle.mean.put(layer, l.add(r));
        else if (l != null) newStyle.mean.put(layer, l);
        else if (r != null) newStyle.mean.put(layer, r);
      });
      Stream.concat(cov0.keySet().stream(), right.cov0.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = cov0.get(layer);
        Tensor r = right.cov0.get(layer);
        if (l != null && l != r) newStyle.cov0.put(layer, l.add(r));
        else if (l != null) newStyle.cov0.put(layer, l);
        else if (r != null) newStyle.cov0.put(layer, r);
      });
      Stream.concat(cov1.keySet().stream(), right.cov1.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = cov1.get(layer);
        Tensor r = right.cov1.get(layer);
        if (l != null && l != r) newStyle.cov1.put(layer, l.add(r));
        else if (l != null) newStyle.cov1.put(layer, l);
        else if (r != null) newStyle.cov1.put(layer, r);
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
   * @param <T> the type parameter
   */
  public static class NeuralSetup<T extends LayerEnum<T>> {
    
    /**
     * The Style parameters.
     */
    public final StyleSetup<T> style;
    /**
     * The Content target.
     */
    public ContentTarget<T> contentTarget = new ContentTarget();
    /**
     * The Style targets.
     */
    public Map<CharSequence, StyleTarget<T>> styleTargets = new HashMap<>();
    
    
    /**
     * Instantiates a new Neural setup.
     *
     * @param style the style
     */
    public NeuralSetup(final StyleSetup style) {this.style = style;}
  }
  
}
