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

package com.simiacryptus.mindseye.app;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer;
import com.simiacryptus.mindseye.layers.cudnn.GateBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.GramianLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.ValueLayer;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.models.MultiLayerImageNetwork;
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
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.Tuple2;

import javax.annotation.Nonnull;
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
 */
public abstract class StyleTransferBase<T extends LayerEnum<T>, U extends MultiLayerImageNetwork<T>> extends ArtistryAppBase {
  
  boolean parallelLossFunctions = true;
  
  public BufferedImage styleTransfer(@Nonnull final NotebookOutput log, final BufferedImage canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes, final NeuralSetup measureStyle) {
    BufferedImage result = canvasImage;
    try {
      PipelineNetwork network = fitnessNetwork(measureStyle);
      log.p("Input Parameters:");
      log.code(() -> {
        return toJson(styleParameters);
      });
      log.p("Input Content:");
      log.p(log.image(styleParameters.contentImage, "Content Image"));
      log.p("Style Content:");
      styleParameters.styleImages.forEach((file, styleImage) -> {
        try {
          log.p(log.image(styleImage, file));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      log.p("Input Canvas:");
      log.p(log.image(canvasImage, "Input Canvas"));
      result = train(log, canvasImage, network, styleParameters.precision, trainingMinutes);
    } catch (Throwable throwable) {
      try {
        log.code(() -> {
          return throwable;
        });
      } catch (Throwable e2) {
      }
    }
    try {
      log.p("Output Canvas:");
      log.p(log.image(canvasImage, "Output Canvas"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return result;
  }
  
  /**
   * Train buffered image.
   *
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param network         the network
   * @param precision       the precision
   * @param trainingMinutes the training minutes
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage train(@Nonnull final NotebookOutput log, final BufferedImage canvasImage, final PipelineNetwork network, final Precision precision, final int trainingMinutes) {
    System.gc();
    Tensor canvas = Tensor.fromRGB(canvasImage);
    TestUtil.monitorImage(canvas, false, false);
    network.setFrozen(true);
    setPrecision(network, precision);
    TestUtil.instrumentPerformance(log, network);
    addLayersHandler(network, server);
    @Nonnull Trainable trainable = new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
    
    log.code(() -> {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      new IterativeTrainer(trainable)
        .setMonitor(TestUtil.getMonitor(history))
//        .setOrientation(new QQN())
        .setOrientation(new TrustRegionStrategy() {
          @Override
          public TrustRegion getRegionPolicy(final Layer layer) {
            return new RangeConstraint().setMin(1e-2).setMax(256);
          }
        })
        .setIterationsPerSample(100)
//        .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
        .setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e6))
//        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
        .setTimeout(trainingMinutes, TimeUnit.MINUTES)
        .setTerminateThreshold(Double.NEGATIVE_INFINITY)
        .runAndFree();
      return TestUtil.plot(history);
    });
    return canvas.toImage();
  }
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
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
    functions.addAll(getContentComponents(setup, nodeMap));
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
          network.wrap(wrapTilesAvg(new GramianLayer()), recentered))
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
  
  @Nonnull
  public abstract T[] getLayerTypes();
  
  /**
   * Gets content components.
   *
   * @param setup   the setup
   * @param nodeMap the node map
   * @return the content components
   */
  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getContentComponents(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> contentComponents = new ArrayList<>();
    for (final T layerType : getLayerTypes()) {
      final DAGNode node = nodeMap.get(layerType);
      final double coeff_content = !setup.style.content.params.containsKey(layerType) ? 0 : setup.style.content.params.get(layerType);
      final PipelineNetwork network1 = (PipelineNetwork) node.getNetwork();
      if (coeff_content != 0) {
        Tensor content = setup.contentTarget.content.get(layerType);
        contentComponents.add(new Tuple2<>(coeff_content, network1.wrap(new MeanSqLossLayer().setAlpha(1.0 / content.rms()),
          node, network1.wrap(new ValueLayer(content), new DAGNode[]{}))));
      }
    }
    return contentComponents;
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
    Tensor contentInput = Tensor.fromRGB(style.contentImage);
    List<Tensor> styleInputs = keyList.stream().map(x -> style.styleImages.get(x)).map(img -> Tensor.fromRGB(img)).collect(Collectors.toList());
    IntStream.range(0, keyList.size()).forEach(i -> {
      self.styleTargets.put(keyList.get(i), new StyleTarget());
    });
    self.contentTarget = new ContentTarget();
    for (final T layerType : getLayerTypes()) {
      System.gc();
      final PipelineNetwork network = layerType.texture();
      setPrecision(network, style.precision);
      Tensor content = network.eval(contentInput).getDataAndFree().getAndFree(0);
      self.contentTarget.content.put(layerType, content);
      logger.info(String.format("%s : target content = %s", layerType.name(), content.prettyPrint()));
      logger.info(String.format("%s : content statistics = %s", layerType.name(), JsonUtil.toJson(new ScalarStatistics().add(content.getData()).getMetrics())));
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        CharSequence key = keyList.get(i);
        StyleTarget<T> styleTarget = self.styleTargets.get(key);
        if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> (LayerStyleParams) x.getValue().params.get(layerType)).filter(x -> null != x).filter(x -> x.mean != 0 || x.cov != 0).count())
          continue;
        System.gc();
        Tensor mean = wrapTilesAvg(avg(network.copy())).eval(styleInput).getDataAndFree().getAndFree(0);
        styleTarget.mean.put(layerType, mean);
        logger.info(String.format("%s : style mean = %s", layerType.name(), mean.prettyPrint()));
        logger.info(String.format("%s : mean statistics = %s", layerType.name(), JsonUtil.toJson(new ScalarStatistics().add(mean.getData()).getMetrics())));
        if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> (LayerStyleParams) x.getValue().params.get(layerType)).filter(x -> null != x).filter(x -> x.cov != 0).count())
          continue;
        System.gc();
        Tensor cov0 = wrapTilesAvg(gram(network.copy())).eval(styleInput).getDataAndFree().getAndFree(0);
        Tensor cov1 = wrapTilesAvg(gram(network.copy(), mean)).eval(styleInput).getDataAndFree().getAndFree(0);
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
  
  protected Layer wrapTilesAvg(final Layer subnet) {
    return wrapTilesAvg(subnet, 0, 0, 0, 0, 400, 400);
  }
  
  protected Layer wrapTilesAvg(final Layer subnet, final int borderX1, final int borderY1, final int borderX2, final int borderY2, final int tileWidth, final int tileHeight) {
    PipelineNetwork network1 = new PipelineNetwork(1);
    if (borderX1 != 0 || borderY1 != 0)
      network1.wrap(new com.simiacryptus.mindseye.layers.cudnn.ImgZeroPaddingLayer(borderX1, borderY1));
    network1.add(subnet);
    if (borderX2 != 0 || borderY2 != 0)
      network1.wrap(new com.simiacryptus.mindseye.layers.cudnn.ImgZeroPaddingLayer(-borderX2, -borderY2));
    PipelineNetwork network = new PipelineNetwork(1);
    network.wrap(new com.simiacryptus.mindseye.layers.cudnn.ImgTileSubnetLayer(network1, tileWidth, tileHeight, tileWidth - 2 * borderX1, tileHeight - 2 * borderY1));
    network.wrap(new BandAvgReducerLayer());
    return network;
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
    setPrecision(network, setup.style.precision);
    return network;
  }
  
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
    functions.stream().filter(x -> x._1 != 0).reduce((a, b) -> {
      return new Tuple2<>(1.0, network.wrap(new BinarySumLayer(a._1, b._1), a._2, b._2).setParallel(parallelLossFunctions));
    }).get();
    return network;
  }
  
  public enum CenteringMode {
    Dynamic,
    Static,
    Origin
  }
  
  /**
   * The type Content coefficients.
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
   */
  public static class StyleSetup<T extends LayerEnum<T>> {
    /**
     * The Precision.
     */
    public final Precision precision;
    /**
     * The Content image.
     */
    public transient BufferedImage contentImage;
    /**
     * The Style image.
     */
    public final transient Map<CharSequence, BufferedImage> styleImages;
    /**
     * The Styles.
     */
    public final Map<List<CharSequence>, StyleCoefficients> styles;
    /**
     * The Content.
     */
    public final ContentCoefficients<T> content;
    
    
    /**
     * Instantiates a new Style setup.
     *
     * @param precision           the precision
     * @param contentImage        the content image
     * @param contentCoefficients the content coefficients
     * @param styleImages         the style image
     * @param styles              the styles
     */
    public StyleSetup(final Precision precision, final BufferedImage contentImage, ContentCoefficients contentCoefficients, final Map<CharSequence, BufferedImage> styleImages, final Map<List<CharSequence>, StyleCoefficients> styles) {
      this.precision = precision;
      this.contentImage = contentImage;
      this.styleImages = styleImages;
      this.styles = styles;
      this.content = contentCoefficients;
    }
    
  }
  
  /**
   * The type Style coefficients.
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
    
    public StyleCoefficients set(final T layerType, final double coeff_style_mean, final double coeff_style_cov) {
      params.put(layerType, new LayerStyleParams(coeff_style_mean, coeff_style_cov));
      return this;
    }
    
  }
  
  /**
   * The type Content target.
   */
  public static class ContentTarget<T extends LayerEnum<T>> {
    /**
     * The Content.
     */
    public Map<T, Tensor> content = new HashMap<>();
  }
  
  /**
   * The type Style target.
   */
  public class StyleTarget<T extends LayerEnum<T>> {
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
   */
  public class NeuralSetup<T extends LayerEnum<T>> {
    
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
