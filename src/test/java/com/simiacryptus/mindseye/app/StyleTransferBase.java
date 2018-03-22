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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
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
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
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

/**
 * This notebook implements the Style Transfer protocol outlined in <a href="https://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a>
 */
public abstract class StyleTransferBase<T extends LayerEnum<T>, U extends MultiLayerImageNetwork<T>> extends ArtistryAppBase {
  
  
  /**
   * Style transfer buffered image.
   *
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage styleTransfer(@Nonnull final NotebookOutput log, final BufferedImage canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes) {
    PipelineNetwork network = fitnessNetwork(measureStyle(styleParameters));
    log.p("Input Parameters:");
    log.code(() -> {
      return toJson(styleParameters);
    });
    try {
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
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    BufferedImage result = train(log, canvasImage, network, styleParameters.precision, trainingMinutes);
    try {
      log.p("Output Canvas:");
      log.p(log.image(result, "Output Canvas"));
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
    @Nonnull Trainable trainable = new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
    TestUtil.instrumentPerformance(log, network);
    addLayersHandler(network, server);
  
    log.code(() -> {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      new IterativeTrainer(trainable)
        .setMonitor(TestUtil.getMonitor(history))
        .setOrientation(new QQN())
//        .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
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
      for (final String key : setup.style.styles.keySet()) {
        StyleTarget<T> t = setup.styleTargets.get(key);
        StyleCoefficients<T> c = setup.style.styles.get(key);
        assert null != c;
        assert null != t;
        final DAGNode node = nodeMap.get(layerType);
        final Tensor target_style_mean = t.mean.get(layerType);
        final Tensor target_style_pca = t.pca.get(layerType);
        final PipelineNetwork network1 = (PipelineNetwork) node.getNetwork();
        if (c.params.containsKey(layerType) && (c.params.get(layerType).cov != 0 || c.params.get(layerType).mean != 0)) {
          DAGNode negTarget = network1.wrap(new ValueLayer(target_style_mean.scale(-1)), new DAGNode[]{});
          InnerNode negAvg = network1.wrap(new BandAvgReducerLayer().setAlpha(-1), node);
          if (c.params.get(layerType).cov != 0) {
            InnerNode recentered;
            if (c.dynamic_center) {
              recentered = network1.wrap(new GateBiasLayer(), node, negAvg);
            }
            else {
              recentered = network1.wrap(new GateBiasLayer(), node, negTarget);
            }
            int[] pcaDim = target_style_pca.getDimensions();
            assert 0 < pcaDim[2] : Arrays.toString(pcaDim);
            int inputBands = target_style_mean.getDimensions()[2];
            assert 0 < inputBands : Arrays.toString(target_style_mean.getDimensions());
            int outputBands = pcaDim[2] / inputBands;
            assert 0 < outputBands : Arrays.toString(pcaDim) + " / " + inputBands;
            styleComponents.add(new Tuple2<>(c.params.get(layerType).cov, network1.wrap(new MeanSqLossLayer(),
              network1.wrap(new ValueLayer(t.pca_cov.get(layerType)), new DAGNode[]{}),
              network1.wrap(new GramianLayer(),
                network1.wrap(new ConvolutionLayer(pcaDim[0], pcaDim[1], inputBands, outputBands).set(target_style_pca),
                  recentered)))
            ));
          }
          if (c.params.get(layerType).mean != 0) {
            styleComponents.add(new Tuple2<>(c.params.get(layerType).mean,
              network1.wrap(new MeanSqLossLayer(), negAvg, negTarget)
            ));
          }
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
        contentComponents.add(new Tuple2<>(coeff_content, network1.wrap(new MeanSqLossLayer(),
          node, network1.wrap(new ValueLayer(setup.contentTarget.content.get(layerType)), new DAGNode[]{}))));
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
    List<String> keyList = style.styleImages.keySet().stream().collect(Collectors.toList());
    Tensor contentInput = Tensor.fromRGB(style.contentImage);
    List<Tensor> styleInputs = keyList.stream().map(x -> style.styleImages.get(x)).map(img -> Tensor.fromRGB(img)).collect(Collectors.toList());
    IntStream.range(0, keyList.size()).forEach(i -> {
      self.styleTargets.put(keyList.get(i), new StyleTarget());
    });
    self.contentTarget = new ContentTarget();
    for (final T layerType : getLayerTypes()) {
      System.gc();
      final PipelineNetwork network = layerType.texture();
      self.contentTarget.content.put(layerType, network.eval(contentInput).getDataAndFree().getAndFree(0));
      logger.info(String.format("target_content_%s=%s", layerType.name(), self.contentTarget.content.get(layerType).prettyPrint()));
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        StyleTarget<T> styleTarget = self.styleTargets.get(keyList.get(i));
        System.gc();
        styleTarget.mean.put(layerType, avg(network.copy()).eval(styleInput).getDataAndFree().getAndFree(0));
        logger.info(String.format("target_style_mean_%s=%s", layerType.name(), styleTarget.mean.get(layerType).prettyPrint()));
        System.gc();
        styleTarget.cov.put(layerType, gram(network.copy(), styleTarget.mean.get(layerType)).eval(styleInput).getDataAndFree().getAndFree(0));
        logger.info(String.format("target_style_cov_%s=%s", layerType.name(), styleTarget.cov.get(layerType).prettyPrint()));
        styleTarget.pca.put(layerType, pca(styleTarget.cov.get(layerType), style.power));
        logger.info(String.format("target_style_pca_%s=%s", layerType.name(), styleTarget.pca.get(layerType).prettyPrint()));
        styleTarget.pca_cov.put(layerType, gram(network.copy(), styleTarget.mean.get(layerType), styleTarget.pca.get(layerType)).eval(styleInput).getDataAndFree().getAndFree(0));
        logger.info(String.format("target_style_pca_cov_%s=%s", layerType.name(), styleTarget.pca_cov.get(layerType).prettyPrint()));
      }
    }
    return self;
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
    PipelineNetwork network = withClamp(measureStyle(setup, nodes, pipelineNetwork));
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
  public PipelineNetwork measureStyle(NeuralSetup setup, final Map<T, DAGNode> nodeMap, final PipelineNetwork network) {
    List<Tuple2<Double, DAGNode>> functions = getFitnessComponents(setup, nodeMap);
    functions.stream().filter(x -> x._1 != 0).reduce((a, b) -> new Tuple2<>(1.0, network.wrap(new BinarySumLayer(a._1, b._1), a._2, b._2))).get();
    return network;
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
    public final transient BufferedImage contentImage;
    /**
     * The Style image.
     */
    public final transient Map<String, BufferedImage> styleImages;
    /**
     * The Styles.
     */
    public final Map<String, StyleCoefficients> styles;
    /**
     * The Content.
     */
    public final ContentCoefficients<T> content;
    /**
     * The Power.
     */
    public double power;
  
  
    /**
     * Instantiates a new Style setup.
     *
     * @param precision           the precision
     * @param contentImage        the content image
     * @param contentCoefficients the content coefficients
     * @param styleImages         the style image
     * @param styles              the styles
     * @param power               the power
     */
    public StyleSetup(final Precision precision, final BufferedImage contentImage, ContentCoefficients contentCoefficients, final Map<String, BufferedImage> styleImages, final Map<String, StyleCoefficients> styles, final double power) {
      this.precision = precision;
      this.contentImage = contentImage;
      this.styleImages = styleImages;
      this.styles = styles;
      this.content = contentCoefficients;
      this.power = power;
    }
    
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
   * The type Style coefficients.
   */
  public static class StyleCoefficients<T extends LayerEnum<T>> {
    /**
     * The Dynamic center.
     */
    public final boolean dynamic_center;
    /**
     * The Params.
     */
    public final Map<T, LayerStyleParams> params = new HashMap<>();
    
    
    /**
     * Instantiates a new Style coefficients.
     *
     * @param dynamicCenter the dynamic center
     */
    public StyleCoefficients(final boolean dynamicCenter) {
      dynamic_center = dynamicCenter;
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
    public Map<T, Tensor> cov = new HashMap<>();
    /**
     * The Mean.
     */
    public Map<T, Tensor> mean = new HashMap<>();
    /**
     * The Pca.
     */
    public Map<T, Tensor> pca = new HashMap<>();
    /**
     * The Pca cov.
     */
    public Map<T, Tensor> pca_cov = new HashMap<>();
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
    public Map<String, StyleTarget<T>> styleTargets = new HashMap<>();
  
  
    /**
     * Instantiates a new Neural setup.
     *
     * @param style the style
     */
    public NeuralSetup(final StyleSetup style) {this.style = style;}
  }
  
}
