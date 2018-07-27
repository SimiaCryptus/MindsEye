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
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.BinarySumLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ValueLayer;
import com.simiacryptus.mindseye.models.CVPipe;
import com.simiacryptus.mindseye.models.CVPipe_VGG16;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.BisectionSearch;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.RangeConstraint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.FileHTTPD;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.io.NullNotebookOutput;
import com.simiacryptus.util.lang.Tuple2;
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

/**
 * This notebook implements the Style Transfer protocol outlined in <a href="https://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a>
 *
 * @param <T> the type parameter
 * @param <U> the type parameter
 */
public abstract class DeepDream<T extends LayerEnum<T>, U extends CVPipe<T>> {
  private static final Logger logger = LoggerFactory.getLogger(DeepDream.class);
  private boolean tiled = false;
  
  /**
   * Deep dream buffered image.
   *
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage deepDream(final BufferedImage canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes) {
    return deepDream(null, new NullNotebookOutput(), canvasImage, styleParameters, trainingMinutes, 50, true);
  }
  
  /**
   * Style transfer buffered image.
   *
   * @param server          the server
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param maxIterations   the max iterations
   * @param verbose         the verbose
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage deepDream(final FileHTTPD server, @Nonnull final NotebookOutput log, final BufferedImage canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes, final int maxIterations, final boolean verbose) {
    PipelineNetwork network = fitnessNetwork(processStats(styleParameters));
    log.p("Input Parameters:");
    log.code(() -> {
      return ArtistryUtil.toJson(styleParameters);
    });
    BufferedImage result = train(server, verbose ? log : new NullNotebookOutput(), canvasImage, network, styleParameters.precision, trainingMinutes, maxIterations);
    log.p("Result:");
    log.p(log.image(result, "Result"));
    return result;
  }
  
  /**
   * Train buffered image.
   *
   * @param server          the server
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param network         the network
   * @param precision       the precision
   * @param trainingMinutes the training minutes
   * @param maxIterations   the max iterations
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage train(final FileHTTPD server, @Nonnull final NotebookOutput log, final BufferedImage canvasImage, PipelineNetwork network, final Precision precision, final int trainingMinutes, final int maxIterations) {
    System.gc();
    Tensor canvas = Tensor.fromRGB(canvasImage);
    String imageName = "image_" + Long.toHexString(MarkdownNotebookOutput.random.nextLong());
    log.p("<a href=\"/" + imageName + ".jpg\"><img src=\"/" + imageName + ".jpg\"></a>");
    log.getHttpd().addHandler(imageName + ".jpg", imageName + "/jpeg", r -> {
      try {
        ImageIO.write(canvas.toImage(), "jpeg", r);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
    TestUtil.monitorImage(canvas, false, false);
    network.setFrozen(true);
    ArtistryUtil.setPrecision(network, precision);
    TestUtil.instrumentPerformance(network);
    if (null != server) ArtistryUtil.addLayersHandler(network, server);
    if (tiled) network = ArtistryUtil.tileCycle(network);
    train(log, network, canvas, trainingMinutes, maxIterations);
    return canvas.toImage();
  }
  
  /**
   * Train.
   *
   * @param log             the log
   * @param network         the network
   * @param canvas          the canvas
   * @param trainingMinutes the training minutes
   * @param maxIterations   the max iterations
   */
  public void train(@Nonnull final NotebookOutput log, final PipelineNetwork network, final Tensor canvas, final int trainingMinutes, final int maxIterations) {
    @Nonnull Trainable trainable = getTrainable(network, canvas);
    @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
    String trainingName = "training_" + Long.toHexString(MarkdownNotebookOutput.random.nextLong());
    log.p("<a href=\"/" + trainingName + ".jpg\"><img src=\"/" + trainingName + ".jpg\"></a>");
    log.getHttpd().addHandler(trainingName + ".jpg", "image/jpeg", r -> {
      try {
        ImageIO.write(Util.toImage(TestUtil.plot(history)), "jpeg", r);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
    log.code(() -> {
      new IterativeTrainer(trainable)
        .setMonitor(TestUtil.getMonitor(history))
        .setIterationsPerSample(100)
        .setOrientation(new TrustRegionStrategy() {
          @Override
          public TrustRegion getRegionPolicy(final com.simiacryptus.mindseye.lang.Layer layer) {
            return new RangeConstraint();
          }
        })
        .setMaxIterations(maxIterations)
        .setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e3))
//        .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
//        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
        .setTimeout(trainingMinutes, TimeUnit.MINUTES)
        .setTerminateThreshold(Double.NEGATIVE_INFINITY)
        .runAndFree();
      return TestUtil.plot(history);
    });
  }
  
  /**
   * Gets trainable.
   *
   * @param network the network
   * @param canvas  the canvas
   * @return the trainable
   */
  @Nonnull
  public Trainable getTrainable(final PipelineNetwork network, final Tensor canvas) {
    return new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
  }
  
  /**
   * Measure style neural setup.
   *
   * @param style the style
   * @return the neural setup
   */
  public NeuralSetup processStats(final StyleSetup<T> style) {
    NeuralSetup<T> self = new NeuralSetup(style);
    Tensor contentInput = Tensor.fromRGB(style.contentImage);
    self.contentTarget = new ContentTarget();
    for (final T layerType : getLayerTypes()) {
      System.gc();
      final PipelineNetwork network = layerType.network();
      ContentCoefficients contentCoefficients = style.coefficients.get(layerType);
      if (null != contentCoefficients && 0 != contentCoefficients.rms) {
        self.contentTarget.content.put(layerType, network.eval(contentInput).getDataAndFree().getAndFree(0));
        logger.info(String.format("target_content_%s=%s", layerType.name(), self.contentTarget.content.get(layerType).prettyPrint()));
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
    PipelineNetwork network = processStats(setup, nodes, pipelineNetwork);
    //network = withClamp(network);
    ArtistryUtil.setPrecision(network, setup.style.precision);
    return network;
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
    return functions;
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
  public ArrayList<Tuple2<Double, DAGNode>> getContentComponents(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> contentComponents = new ArrayList<>();
    for (final T layerType : getLayerTypes()) {
      final DAGNode node = nodeMap.get(layerType);
      if (setup.style.coefficients.containsKey(layerType)) {
        DAGNetwork network = node.getNetwork();
        final double coeff_content = setup.style.coefficients.get(layerType).rms;
        if (0 != coeff_content) {
          Tensor contentSignal = setup.contentTarget.content.get(layerType);
          if (contentSignal != null) {
            contentComponents.add(new Tuple2<>(coeff_content, network.wrap(new MeanSqLossLayer(),
              node, network.wrap(new ValueLayer(contentSignal)))));
          }
          else {
            logger.info("No content signal for " + layerType);
          }
        }
        final double coeff_gain = setup.style.coefficients.get(layerType).gain;
        if (0 != coeff_gain) {
          contentComponents.add(new Tuple2<>(-coeff_gain, network.wrap(new AvgReducerLayer(),
            network.wrap(new SquareActivationLayer(), node))));
        }
      }
    }
    return contentComponents;
  }
  
  /**
   * Measure style pipeline network.
   *
   * @param setup   the setup
   * @param nodeMap the node buildMap
   * @param network the network
   * @return the pipeline network
   */
  public PipelineNetwork processStats(NeuralSetup setup, final Map<T, DAGNode> nodeMap, final PipelineNetwork network) {
    List<Tuple2<Double, DAGNode>> functions = getFitnessComponents(setup, nodeMap);
    functions.stream().filter(x -> x._1 != 0).reduce((a, b) -> new Tuple2<>(1.0, network.wrap(new BinarySumLayer(a._1, b._1), a._2, b._2))).get();
    return network;
  }
  
  /**
   * Gets instance.
   *
   * @return the instance
   */
  public abstract U getInstance();
  
  /**
   * Is tiled boolean.
   *
   * @return the boolean
   */
  public boolean isTiled() {
    return tiled;
  }
  
  /**
   * Sets tiled.
   *
   * @param tiled the tiled
   * @return the tiled
   */
  public DeepDream<T, U> setTiled(boolean tiled) {
    this.tiled = tiled;
    return this;
  }
  
  /**
   * The type Vgg 16.
   */
  public static class VGG16 extends DeepDream<CVPipe_VGG16.Layer, CVPipe_VGG16> {
  
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
  public static class VGG19 extends DeepDream<CVPipe_VGG19.Layer, CVPipe_VGG19> {
  
    public CVPipe_VGG19 getInstance() {
      return CVPipe_VGG19.INSTANCE;
    }
    
    @Nonnull
    public CVPipe_VGG19.Layer[] getLayerTypes() {
      return CVPipe_VGG19.Layer.values();
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
     * The Content image.
     */
    public final transient BufferedImage contentImage;
    /**
     * The Content.
     */
    public final Map<T, ContentCoefficients> coefficients;
  
  
    /**
     * Instantiates a new Style setup.
     *
     * @param precision           the precision
     * @param contentImage        the content image
     * @param contentCoefficients the content coefficients
     */
    public StyleSetup(final Precision precision, final BufferedImage contentImage, Map<T, ContentCoefficients> contentCoefficients) {
      this.precision = precision;
      this.contentImage = contentImage;
      this.coefficients = contentCoefficients;
    }
    
  }
  
  /**
   * The type Content coefficients.
   */
  public static class ContentCoefficients {
    /**
     * The Rms.
     */
    public final double rms;
    /**
     * The Gain.
     */
    public final double gain;
  
    /**
     * Instantiates a new Content coefficients.
     *
     * @param rms  the rms
     * @param gain the gain
     */
    public ContentCoefficients(final double rms, final double gain) {
      this.rms = rms;
      this.gain = gain;
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
   * The type Neural setup.
   *
   * @param <T> the type parameter
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
     * Instantiates a new Neural setup.
     *
     * @param style the style
     */
    public NeuralSetup(final StyleSetup style) {this.style = style;}
  }
  
}
