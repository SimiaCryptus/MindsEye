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
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.ValueLayer;
import com.simiacryptus.mindseye.models.CVPipe;
import com.simiacryptus.mindseye.models.CVPipe_VGG16;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.network.DAGNetwork;
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
import com.simiacryptus.util.FileNanoHTTPD;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.ScalarStatistics;
import com.simiacryptus.util.io.JsonUtil;
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
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * This notebook implements the Style Transfer protocol outlined in <a href="https://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a>
 *
 * @param <T> the type parameter
 * @param <U> the type parameter
 */
public abstract class SegmentedStyleTransfer<T extends LayerEnum<T>, U extends CVPipe<T>> {
  
  private static final Logger logger = LoggerFactory.getLogger(SegmentedStyleTransfer.class);
  /**
   * The Parallel loss functions.
   */
  public boolean parallelLossFunctions = true;
  private boolean tiled = false;
  private int content_masks = 3;
  private int content_colorClusters = 3;
  private int content_textureClusters = 3;
  private int style_masks = 3;
  private int stlye_colorClusters = 3;
  private int style_textureClusters = 3;
  
  /**
   * Alpha list list.
   *
   * @param styleInput the style input
   * @param tensors    the tensors
   * @return the list
   */
  public static List<Tensor> alphaList(final Tensor styleInput, final List<Tensor> tensors) {
    return tensors.stream().map(x -> alpha(styleInput, x)).collect(Collectors.toList());
  }
  
  /**
   * Alpha map map.
   *
   * @param styleInput the style input
   * @param tensors    the tensors
   * @return the map
   */
  public static Map<Tensor, Tensor> alphaMap(final Tensor styleInput, final List<Tensor> tensors) {
    return tensors.stream().collect(Collectors.toMap(x -> x, x -> alpha(styleInput, x)));
  }
  
  /**
   * Alpha tensor.
   *
   * @param content the content
   * @param mask    the mask
   * @return the tensor
   */
  public static Tensor alpha(final Tensor content, final Tensor mask) {
    int xbands = mask.getDimensions()[2] - 1;
    return content.mapCoords(c -> {
      int[] coords = c.getCoords();
      return content.get(c) * mask.get(coords[0], coords[1], Math.min(coords[2], xbands));
    });
  }
  
  public static double alphaMaskSimilarity(final Tensor contentMask, final Tensor styleMask) {
    Tensor l = contentMask.sumChannels();
    Tensor r = styleMask.sumChannels();
    int[] dimensions = r.getDimensions();
    Tensor resize = Tensor.fromRGB(TestUtil.resize(l.toImage(), dimensions[0], dimensions[1])).sumChannels();
    Tensor a = resize.unit();
    Tensor b = r.unit();
    double dot = a.dot(b);
    a.freeRef();
    b.freeRef();
    r.freeRef();
    l.freeRef();
    return dot;
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
  public BufferedImage transfer(final BufferedImage canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes, final NeuralSetup measureStyle) {
    return transfer(null, new NullNotebookOutput(), canvasImage, styleParameters, trainingMinutes, measureStyle, 50, true);
  }
  
  /**
   * Transfer tensor.
   *
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param measureStyle    the measure style
   * @return the tensor
   */
  public Tensor transfer(final Tensor canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes, final NeuralSetup measureStyle) {
    return transfer(null, new NullNotebookOutput(), canvasImage, styleParameters, trainingMinutes, measureStyle, 50, true);
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
   * @param maxIterations   the max iterations
   * @param verbose         the verbose
   * @return the buffered image
   */
  public BufferedImage transfer(final FileNanoHTTPD server, @Nonnull final NotebookOutput log, final BufferedImage canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes, final NeuralSetup measureStyle, final int maxIterations, final boolean verbose) {
    try {
      Tensor canvas = Tensor.fromRGB(canvasImage);
      transfer(server, log, styleParameters, trainingMinutes, measureStyle, maxIterations, verbose, canvas);
      BufferedImage result = log.code(() -> {
        return canvas.toImage();
      });
      log.p("Result:");
      log.p(log.image(result, "Output Canvas"));
      return result;
    } catch (Throwable e) {
      return canvasImage;
    }
  }
  
  /**
   * Transfer tensor.
   *
   * @param server          the server
   * @param log             the log
   * @param canvasData      the canvas data
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param measureStyle    the measure style
   * @param maxIterations   the max iterations
   * @param verbose         the verbose
   * @return the tensor
   */
  public Tensor transfer(final FileNanoHTTPD server, @Nonnull final NotebookOutput log, final Tensor canvasData, final StyleSetup<T> styleParameters, final int trainingMinutes, final NeuralSetup measureStyle, final int maxIterations, final boolean verbose) {
    try {
      transfer(server, log, styleParameters, trainingMinutes, measureStyle, maxIterations, verbose, canvasData);
      log.p("Result:");
      log.p(log.image(canvasData.toImage(), "Output Canvas"));
      return canvasData;
    } catch (Throwable e) {
      return canvasData;
    }
  }
  
  /**
   * Transfer.
   *
   * @param server          the server
   * @param log             the log
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param measureStyle    the measure style
   * @param maxIterations   the max iterations
   * @param verbose         the verbose
   * @param canvas          the canvas
   */
  public void transfer(final FileNanoHTTPD server, @Nonnull final NotebookOutput log, final StyleSetup<T> styleParameters, final int trainingMinutes, final NeuralSetup measureStyle, final int maxIterations, final boolean verbose, final Tensor canvas) {
//      log.p("Input Content:");
//      log.p(log.image(styleParameters.contentImage, "Content Image"));
//      log.p("Style Content:");
//      styleParameters.styleImages.forEach((file, styleImage) -> {
//        log.p(log.image(styleImage, file));
//      });
//      log.p("Input Canvas:");
//      log.p(log.image(canvasImage, "Input Canvas"));
    System.gc();
    NotebookOutput trainingLog = verbose ? log : new NullNotebookOutput();
    log.h2("Content Partitioning");
    List<Tensor> masks = ImageSegmenter.quickMasks(log, measureStyle.contentSource, getContent_masks(), getContent_colorClusters(), getContent_textureClusters());
    System.gc();
    log.h2("Content Painting");
    TestUtil.monitorImage(canvas, false, false);
    log.p("<a href=\"/image.jpg\"><img src=\"/image.jpg\"></a>");
    log.getHttpd().addHandler("image.jpg", "image/jpeg", r -> {
      try {
        ImageIO.write(canvas.toImage(), "jpeg", r);
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
    Trainable trainable = trainingLog.code(() -> {
      PipelineNetwork network = fitnessNetwork(measureStyle, masks);
      network.setFrozen(true);
      if (null != server) ArtistryUtil.addLayersHandler(network, server);
      if (tiled) network = ArtistryUtil.tileCycle(network);
      ArtistryUtil.setPrecision(network, styleParameters.precision);
      TestUtil.instrumentPerformance(network);
      Trainable trainable1 = getTrainable(canvas, network);
      network.freeRef();
      return trainable1;
    });
    masks.forEach(ReferenceCountingBase::freeRef);
    try {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      log.p("<a href=\"/training.jpg\"><img src=\"/training.jpg\"></a>");
      log.getHttpd().addHandler("training.jpg", "image/jpeg", r -> {
        try {
          ImageIO.write(Util.toImage(TestUtil.plot(history)), "jpeg", r);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      trainingLog.code(() -> {
        new IterativeTrainer(trainable)
          .setMonitor(TestUtil.getMonitor(history))
          .setOrientation(new TrustRegionStrategy() {
            @Override
            public TrustRegion getRegionPolicy(final Layer layer) {
              return new RangeConstraint().setMin(1e-2).setMax(256);
            }
          })
          .setMaxIterations(maxIterations)
          .setIterationsPerSample(100)
          .setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e6))
          .setTimeout(trainingMinutes, TimeUnit.MINUTES)
          .setTerminateThreshold(Double.NEGATIVE_INFINITY)
          .runAndFree();
        return TestUtil.plot(history);
      });
    } finally {
      trainable.freeRef();
    }
  }
  
  /**
   * Gets trainable.
   *
   * @param canvas  the canvas
   * @param network the network
   * @return the trainable
   */
  @Nonnull
  public Trainable getTrainable(final Tensor canvas, final PipelineNetwork network) {
    return new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
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
      node.addRef();
      InnerNode negAvg = network.wrap(new BandAvgReducerLayer().setAlpha(-1), node);
      if (styleParams.enhance != 0 || styleParams.cov != 0) {
        DAGNode recentered;
        switch (centeringMode) {
          case Origin:
            node.addRef();
            recentered = node;
            break;
          case Dynamic:
            negAvg.addRef();
            node.addRef();
            recentered = network.wrap(new GateBiasLayer(), node, negAvg);
            break;
          case Static:
            node.addRef();
            negTarget.addRef();
            recentered = network.wrap(new GateBiasLayer(), node, negTarget);
            break;
          default:
            throw new RuntimeException();
        }
        int[] covDim = covariance.getDimensions();
        double covRms = covariance.rms();
        if (styleParams.enhance != 0) {
          recentered.addRef();
          styleComponents.add(new Tuple2<>(-(0 == covRms ? styleParams.enhance : (styleParams.enhance / covRms)), network.wrap(new AvgReducerLayer(),
            network.wrap(new SquareActivationLayer(), recentered))));
        }
        if (styleParams.cov != 0) {
          assert 0 < covDim[2] : Arrays.toString(covDim);
          int inputBands = mean.getDimensions()[2];
          assert 0 < inputBands : Arrays.toString(mean.getDimensions());
          int outputBands = covDim[2] / inputBands;
          assert 0 < outputBands : Arrays.toString(covDim) + " / " + inputBands;
          double covScale = 0 == covRms ? 1 : (1.0 / covRms);
          recentered.addRef();
          styleComponents.add(new Tuple2<>(styleParams.cov, network.wrap(new MeanSqLossLayer().setAlpha(covScale),
            network.wrap(new ValueLayer(covariance), new DAGNode[]{}),
            network.wrap(new GramianLayer(), recentered))
          ));
        }
        recentered.freeRef();
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
   * @param log   the log
   * @param style the style
   * @return the neural setup
   */
  public NeuralSetup<T> measureStyle(final NotebookOutput log, final StyleSetup<T> style) {
    NeuralSetup<T> self = new NeuralSetup(style);
    List<CharSequence> keyList = style.styleImages.keySet().stream().collect(Collectors.toList());
    Tensor contentInput = Tensor.fromRGB(style.contentImage);
    List<Tensor> styleInputs = keyList.stream().map(x -> style.styleImages.get(x)).map(img -> Tensor.fromRGB(img)).collect(Collectors.toList());
    log.h2("Style Partitioning");
    Map<Tensor, List<Tensor>> masks = styleInputs.stream().collect(Collectors.toMap(x -> x, (Tensor styleInput) -> {
      return ImageSegmenter.quickMasks(log, styleInput, getStyle_masks(), getStlye_colorClusters(), getStyle_textureClusters());
    }));
    log.h2("Style Measurement");
    IntStream.range(0, keyList.size()).forEach(i -> {
      self.styleTargets.put(keyList.get(i), new SegmentedStyleTarget());
    });
    self.contentTarget = new ContentTarget();
    self.contentSource = contentInput;
    for (final T layerType : getLayerTypes()) {
      System.gc();
      Layer network = layerType.network();
      try {
        ArtistryUtil.setPrecision((DAGNetwork) network, style.precision);
        //network = new ImgTileSubnetLayer(network, 400,400,400,400);
        Tensor content = network.eval(contentInput).getDataAndFree().getAndFree(0);
        self.contentTarget.content.put(layerType, content);
        logger.info(String.format("%s : target content = %s", layerType.name(), content.prettyPrint()));
        logger.info(String.format("%s : content statistics = %s", layerType.name(), JsonUtil.toJson(new ScalarStatistics().add(content.getData()).getMetrics())));
        for (int i = 0; i < styleInputs.size(); i++) {
          CharSequence key = keyList.get(i);
          SegmentedStyleTarget<T> segmentedStyleTarget = self.styleTargets.get(key);
          if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> (LayerStyleParams) x.getValue().params.get(layerType)).filter(x -> null != x).filter(x -> x.mean != 0 || x.cov != 0).count())
            continue;
          System.gc();
          
          Tensor styleInput = styleInputs.get(i);
          alphaMap(styleInput, masks.get(styleInput)).forEach((mask, styleMask) -> {
            StyleTarget<T> styleTarget = segmentedStyleTarget.getSegment(mask);
            Layer wrapAvg = ArtistryUtil.wrapTiledAvg(network.copy(), 400);
            Tensor mean = null;
            try {
              mean = wrapAvg.eval(styleMask).getDataAndFree().getAndFree(0);
              if (styleTarget.mean.put(layerType, mean) != null) throw new AssertionError();
              logger.info(String.format("%s : style mean = %s", layerType.name(), mean.prettyPrint()));
              logger.info(String.format("%s : mean statistics = %s", layerType.name(), JsonUtil.toJson(new ScalarStatistics().add(mean.getData()).getMetrics())));
              if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> (LayerStyleParams) x.getValue().params.get(layerType)).filter(x -> null != x).filter(x -> x.cov != 0).count())
                return;
              System.gc();
              Layer gram = null;
              Tensor cov0;
              try {
                gram = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy()), 400);
                cov0 = gram.eval(styleMask).getDataAndFree().getAndFree(0);
              } finally {
                gram.freeRef();
              }
              Tensor cov1;
              try {
                gram = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy(), mean), 400);
                cov1 = gram.eval(styleMask).getDataAndFree().getAndFree(0);
              } finally {
                gram.freeRef();
              }
              styleMask.freeRef();
              if (styleTarget.cov0.put(layerType, cov0) != null) throw new AssertionError();
              if (styleTarget.cov1.put(layerType, cov1) != null) throw new AssertionError();
              int featureBands = mean.getDimensions()[2];
              int covarianceElements = cov1.getDimensions()[2];
              int selectedBands = covarianceElements / featureBands;
              logger.info(String.format("%s : target cov0 = %s", layerType.name(), cov0.reshapeCast(featureBands, selectedBands, 1).prettyPrintAndFree()));
              logger.info(String.format("%s : cov0 statistics = %s", layerType.name(), JsonUtil.toJson(new ScalarStatistics().add(cov0.getData()).getMetrics())));
              logger.info(String.format("%s : target cov1 = %s", layerType.name(), cov1.reshapeCast(featureBands, selectedBands, 1).prettyPrintAndFree()));
              logger.info(String.format("%s : cov1 statistics = %s", layerType.name(), JsonUtil.toJson(new ScalarStatistics().add(cov1.getData()).getMetrics())));
            } finally {
              styleTarget.freeRef();
              wrapAvg.freeRef();
              if (mean != null) mean.freeRef();
            }
          });
        }
      } finally {
        network.freeRef();
      }
    }
    masks.forEach((k, v) -> {
      k.freeRef();
      v.forEach(ReferenceCountingBase::freeRef);
    });
    return self;
  }
  
  /**
   * Gets style components.
   *
   * @param setup    the setup
   * @param nodeMap  the node buildMap
   * @param selector the selector
   * @return the style components
   */
  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap, final Function<SegmentedStyleTarget<T>, StyleTarget<T>> selector) {
    ArrayList<Tuple2<Double, DAGNode>> styleComponents = new ArrayList<>();
    for (final List<CharSequence> keys : setup.style.styles.keySet()) {
      StyleTarget<T> styleTarget = keys.stream().map(x -> {
        SegmentedStyleTarget<T> obj = setup.styleTargets.get(x);
        StyleTarget<T> choose = selector.apply(obj);
        choose.addRef();
        return choose;
      }).reduce((a, b) -> {
        StyleTarget<T> r = a.add(b);
        a.freeRef();
        b.freeRef();
        return r;
      }).map(x -> {
        StyleTarget<T> r = x.scale(1.0 / keys.size());
        x.freeRef();
        return r;
      }).get();
      assert null != styleTarget;
      for (final T layerType : getLayerTypes()) {
        final StyleCoefficients<T> styleCoefficients = setup.style.styles.get(keys);
        assert null != styleCoefficients;
        styleComponents.addAll(getStyleComponents(nodeMap, layerType, styleCoefficients, styleTarget));
      }
      styleTarget.freeRef();
    }
    return styleComponents;
  }
  
  /**
   * Gets style components.
   *
   * @param nodeMap            the node map
   * @param layerType          the layer type
   * @param styleCoefficients  the style coefficients
   * @param chooseStyleSegment the choose style segment
   * @return the style components
   */
  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(final Map<T, DAGNode> nodeMap, final T layerType, final StyleCoefficients<T> styleCoefficients, final StyleTarget<T> chooseStyleSegment) {
    final DAGNode node = nodeMap.get(layerType);
    if (null == node) throw new RuntimeException("Not Found: " + layerType);
    final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
    LayerStyleParams styleParams = styleCoefficients.params.get(layerType);
    Tensor mean = chooseStyleSegment.mean.get(layerType);
    Tensor covariance;
    switch (styleCoefficients.centeringMode) {
      case Origin:
        covariance = chooseStyleSegment.cov0.get(layerType);
        break;
      case Dynamic:
      case Static:
        covariance = chooseStyleSegment.cov1.get(layerType);
        break;
      default:
        throw new RuntimeException();
    }
    return getStyleComponents(node, network, styleParams, mean, covariance, styleCoefficients.centeringMode);
  }
  
  @Nonnull
  public PipelineNetwork fitnessNetwork(final NeuralSetup setup, final List<Tensor> masks) {
    U networkModel = getNetworkModel();
    PipelineNetwork mainNetwork = networkModel.getNetwork();
    Map<T, UUID> modelNodes = networkModel.getNodes();
    List<Tuple2<Double, DAGNode>> mainFunctions = new ArrayList<>();
    Map<T, DAGNode> mainNodes = getNodes(modelNodes, mainNetwork, null);
    mainFunctions.addAll(getContentComponents(setup, mainNodes));
    masks.forEach((contentMask) -> {
      HashMap<String, String> idMap = new HashMap<>();
      DAGNetwork branchNetwork = mainNetwork.scrambleCopy(idMap);
      //logger.info("Branch Keys");
      //branchNetwork.logKeys();
      Map<T, DAGNode> nodeMap = getNodes(modelNodes, branchNetwork, idMap);
      List<Tuple2<Double, DAGNode>> branchFunctions = new ArrayList<>();
      branchFunctions.addAll(getStyleComponents(setup, nodeMap,
        x -> x.segments.entrySet().stream().max(Comparator.comparingDouble(e -> alphaMaskSimilarity(contentMask, e.getKey()))).get().getValue()));
      ArtistryUtil.reduce(branchNetwork, branchFunctions, parallelLossFunctions);
      InnerNode importNode = mainNetwork.wrap(branchNetwork,
        mainNetwork.wrap(new ProductLayer(), mainNetwork.getInput(0), mainNetwork.constValue(contentMask))
      );
      mainFunctions.add(new Tuple2<>(1.0, importNode));
    });
    ArtistryUtil.reduce(mainNetwork, mainFunctions, parallelLossFunctions);
    ArtistryUtil.setPrecision(mainNetwork, setup.style.precision);
    return mainNetwork;
  }
  
  @Nonnull
  public Map<T, DAGNode> getNodes(final Map<T, UUID> modelNodes, final DAGNetwork network, final HashMap<String, String> replacements) {
    Map<T, DAGNode> nodes = new HashMap<>();
    modelNodes.forEach((l, id) -> {
      UUID replaced = null == replacements ? id : replace(replacements, id);
      DAGNode childNode = network.getChildNode(replaced);
      if (null == childNode) {
        logger.warn(String.format("Could not find Node ID %s (replaced from %s) to represent %s", replaced, id, l));
      }
      else {
        nodes.put(l, childNode);
      }
    });
    return nodes;
  }
  
  @Nonnull
  public UUID replace(final HashMap<String, String> replacements, final UUID id) {
    return UUID.fromString(replacements.getOrDefault(id.toString(), id.toString()));
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
   * Gets instance.
   *
   * @return the instance
   */
  public abstract U getNetworkModel();
  
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
  public SegmentedStyleTransfer<T, U> setTiled(boolean tiled) {
    this.tiled = tiled;
    return this;
  }
  
  public int getContent_masks() {
    return content_masks;
  }
  
  public SegmentedStyleTransfer<T, U> setContent_masks(int content_masks) {
    this.content_masks = content_masks;
    return this;
  }
  
  public int getContent_colorClusters() {
    return content_colorClusters;
  }
  
  public SegmentedStyleTransfer<T, U> setContent_colorClusters(int content_colorClusters) {
    this.content_colorClusters = content_colorClusters;
    return this;
  }
  
  public int getContent_textureClusters() {
    return content_textureClusters;
  }
  
  public SegmentedStyleTransfer<T, U> setContent_textureClusters(int content_textureClusters) {
    this.content_textureClusters = content_textureClusters;
    return this;
  }
  
  public int getStyle_masks() {
    return style_masks;
  }
  
  public SegmentedStyleTransfer<T, U> setStyle_masks(int style_masks) {
    this.style_masks = style_masks;
    return this;
  }
  
  public int getStlye_colorClusters() {
    return stlye_colorClusters;
  }
  
  public SegmentedStyleTransfer<T, U> setStlye_colorClusters(int stlye_colorClusters) {
    this.stlye_colorClusters = stlye_colorClusters;
    return this;
  }
  
  public int getStyle_textureClusters() {
    return style_textureClusters;
  }
  
  public SegmentedStyleTransfer<T, U> setStyle_textureClusters(int style_textureClusters) {
    this.style_textureClusters = style_textureClusters;
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
  public static class VGG16 extends SegmentedStyleTransfer<CVPipe_VGG16.Layer, CVPipe_VGG16> {
    
    public CVPipe_VGG16 getNetworkModel() {
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
  public static class VGG19 extends SegmentedStyleTransfer<CVPipe_VGG19.Layer, CVPipe_VGG19> {
    
    public CVPipe_VGG19 getNetworkModel() {
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
     * The Content image.
     */
    public transient BufferedImage contentImage;
    
    
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
    public StyleCoefficients set(final T layerType, final double coeff_style_mean, final double coeff_style_cov) {return set(layerType, coeff_style_mean, coeff_style_cov, 0.0);}
    
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
   * The type Segmented style target.
   *
   * @param <T> the type parameter
   */
  public class SegmentedStyleTarget<T extends LayerEnum<T>> {
    /**
     * The Segments.
     */
    private final Map<Tensor, StyleTarget<T>> segments = new HashMap<>();
    
    /**
     * Gets segment.
     *
     * @param styleMask the style mask
     * @return the segment
     */
    public StyleTarget<T> getSegment(final Tensor styleMask) {
      synchronized (segments) {
        StyleTarget<T> styleTarget = segments.computeIfAbsent(styleMask, x -> {
          StyleTarget<T> tStyleTarget = new StyleTarget<>();
          styleMask.addRef();
          return tStyleTarget;
        });
        styleTarget.addRef();
        return styleTarget;
      }
    }
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
     * The Style targets.
     */
    public Map<CharSequence, SegmentedStyleTarget<T>> styleTargets = new HashMap<>();
    /**
     * The Content source.
     */
    public Tensor contentSource;
    
    
    /**
     * Instantiates a new Neural setup.
     *
     * @param style the style
     */
    public NeuralSetup(final StyleSetup style) {this.style = style;}
  }
  
}
