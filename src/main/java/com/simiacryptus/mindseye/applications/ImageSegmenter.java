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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.MutableResult;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandSelectLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer;
import com.simiacryptus.mindseye.layers.java.SumReducerLayer;
import com.simiacryptus.mindseye.models.CVPipe;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.io.NullNotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Object location.
 *
 * @param <T> the type parameter
 * @param <U> the type parameter
 */
public abstract class ImageSegmenter<T extends LayerEnum<T>, U extends CVPipe<T>> extends PixelClusterer<T, U> {
  
  private static final Logger logger = LoggerFactory.getLogger(ImageSegmenter.class);
  
  /**
   * Instantiates a new Image segmenter.
   *
   * @param clusters                   the clusters
   * @param orientation                the orientation
   * @param globalDistributionEmphasis the global distribution emphasis
   * @param selectionEntropyAdj        the selection entropy adj
   * @param maxIterations              the max iterations
   * @param timeoutMinutes             the timeout minutes
   * @param seedPcaPower               the seed pca power
   * @param seedMagnitude              the seed magnitude
   */
  public ImageSegmenter(final int clusters, final int orientation, final double globalDistributionEmphasis, final double selectionEntropyAdj, final int maxIterations, final int timeoutMinutes, final double seedPcaPower, final double seedMagnitude) {
    super(clusters, orientation, globalDistributionEmphasis, selectionEntropyAdj, maxIterations, timeoutMinutes, seedPcaPower, seedMagnitude, false, true, 0.0, 1.0, new double[]{1e-1, 1e-3});
  }
  
  /**
   * Instantiates a new Image segmenter.
   *
   * @param clusters the clusters
   */
  public ImageSegmenter(final int clusters) {
    super(clusters);
  }
  
  /**
   * Quick masks list.
   *
   * @param img the img
   * @return the list
   */
  public static List<Tensor> quickMasks(final Tensor img) {
    return quickMasks(new NullNotebookOutput(), img);
  }
  
  /**
   * Quick masks list.
   *
   * @param log the log
   * @param img the img
   * @return the list
   */
  public static List<Tensor> quickMasks(@Nonnull final NotebookOutput log, final Tensor img) {
    return quickMasks(log, img, 9, 3, CVPipe_VGG19.Layer.Layer_0,
      CVPipe_VGG19.Layer.Layer_1a,
      CVPipe_VGG19.Layer.Layer_1e);
  }
  
  /**
   * Quick masks list.
   *
   * @param log      the log
   * @param img      the img
   * @param blur     the blur
   * @param clusters the clusters
   * @param layers   the layers
   * @return the list
   */
  public static List<Tensor> quickMasks(@Nonnull final NotebookOutput log, final Tensor img, final int blur, final int clusters, final LayerEnum... layers) {
    ImageSegmenter segmenter = new ImageSegmenter.VGG19(clusters) {
      @Override
      public Layer modelingNetwork(final CVPipe_VGG19.Layer layer, final Tensor metrics) {
        if (layer == CVPipe_VGG19.Layer.Layer_0) {
          return modelingNetwork(getGlobalBias(), getGlobalGain(), metrics, true, isRescale(), getClusters(), getSeedMagnitude(), 0);
        }
        else {
          return modelingNetwork(getGlobalBias(), getGlobalGain(), metrics, isRecenter(), isRescale(), getClusters(), getSeedMagnitude(), getSeedPcaPower());
        }
      }
    };
    return segmenter.spatialClusters(log, img, PCAObjectLocation.blur(segmenter.featureClusters(log, img, layers), blur));
  }
  
  /**
   * Alpha image mask buffered image.
   *
   * @param log  the log
   * @param img  the img
   * @param mask the mask
   * @return the buffered image
   */
  public static BufferedImage alphaImageMask(@Nonnull final NotebookOutput log, final Tensor img, Tensor mask) {
    return log.code(() -> {
      return img.mapCoords(c -> img.get(c) * mask.get(c.getCoords()[0], c.getCoords()[1], Math.min(c.getCoords()[2], mask.getDimensions()[2]))).toImage();
    });
  }
  
  /**
   * Alpha image mask buffered image.
   *
   * @param img  the img
   * @param mask the mask
   * @return the buffered image
   */
  public static BufferedImage alphaImageMask(final Tensor img, Tensor mask) {
    return img.mapCoords(c -> img.get(c) * mask.get(c.getCoords()[0], c.getCoords()[1], Math.min(c.getCoords()[2], mask.getDimensions()[2] - 1))).toImage();
  }
  
  /**
   * Display image mask.
   *
   * @param log  the log
   * @param img  the img
   * @param mask the mask
   */
  public static void displayImageMask(@Nonnull final NotebookOutput log, final Tensor img, Tensor mask) {
    log.p(log.image(img.toRgbImageAlphaMask(0, 1, 2, mask.scale(255.0)), "") +
      log.image(img.toRgbImageAlphaMask(0, 1, 2, mask.normalizeDistribution().scale(255.0)), ""));
  }
  
  /**
   * Feature clusters list.
   *
   * @param log    the log
   * @param img    the img
   * @param layers the layers
   * @return the list
   */
  public List<Tensor> featureClusters(@Nonnull final NotebookOutput log, final Tensor img, final T... layers) {
    return Arrays.stream(getLayerTypes()).filter(x -> Arrays.asList(layers).contains(x)).flatMap(layer -> {
      log.h2(layer.name());
      Layer network = getInstance().getPrototypes().get(layer);
      ArtistryUtil.setPrecision((DAGNetwork) network, Precision.Float);
      network.setFrozen(true);
      Result imageFeatures = network.eval(new MutableResult(img));
      Tensor featureImage = imageFeatures.getData().get(0);
      log.p("Feature Image Dimension: " + Arrays.toString(featureImage.getDimensions()));
      Layer analyze1 = analyze(layer, log, featureImage);
      List<Tensor> layerMasks = IntStream.range(0, getClusters()).mapToObj(i -> {
        try {
          Tensor maskData = new Tensor(((Layer) PipelineNetwork.build(1,
            analyze1,
            new ImgBandSelectLayer(i, i + 1)
          )).copy().freeze().andThen(new SumReducerLayer()).eval(imageFeatures).getSingleDelta(), img.getDimensions()).map(v -> Math.abs(v));
          displayImageMask(log, img, maskData.sumChannels().rescaleRms(1.0));
          return maskData;
        } catch (Throwable e) {
          logger.warn("Error", e);
          return null;
        }
      }).filter(x -> x != null).collect(Collectors.toList());
      log.p(TestUtil.animatedGif(log, layerMasks.stream().map(selectedBand -> alphaImageMask(img, selectedBand.rescaleRms(1.0))).toArray(i -> new BufferedImage[i])));
      return layerMasks.stream();
    }).collect(Collectors.toList());
  }
  
  /**
   * Spatial clusters list.
   *
   * @param log          the log
   * @param img          the img
   * @param featureMasks the feature masks
   * @return the list
   */
  public List<Tensor> spatialClusters(@Nonnull final NotebookOutput log, final Tensor img, final List<Tensor> featureMasks) {
    Tensor concat = ImgConcatLayer.eval(featureMasks.stream().map(Tensor::sumChannels).collect(Collectors.toList()));
    Tensor reclustered = analyze(null, log, concat).eval(concat).getDataAndFree().getAndFree(0);
    List<Tensor> tensorList = IntStream.range(0, reclustered.getDimensions()[2]).mapToObj(i -> reclustered.selectBand(i)).collect(Collectors.toList());
    log.p(TestUtil.animatedGif(log, tensorList.stream().map(selectedBand -> alphaImageMask(img, selectedBand)).toArray(i -> new BufferedImage[i])));
    for (Tensor selectBand : tensorList) {
      displayImageMask(log, img, selectBand);
    }
    return tensorList;
  }
  
  /**
   * Gets instance.
   *
   * @return the instance
   */
  public abstract U getInstance();
  
  /**
   * Get layer types t [ ].
   *
   * @return the t [ ]
   */
  @Nonnull
  public abstract T[] getLayerTypes();
  
  /**
   * The type Vgg 19.
   */
  public static class VGG19 extends ImageSegmenter<CVPipe_VGG19.Layer, CVPipe_VGG19> {
  
    /**
     * Instantiates a new Vgg 19.
     *
     * @param clusters the clusters
     */
    public VGG19(final int clusters) {
      super(clusters);
    }
  
    /**
     * Instantiates a new Vgg 19.
     *
     * @param clusters                   the clusters
     * @param orientation                the orientation
     * @param globalDistributionEmphasis the global distribution emphasis
     * @param selectionEntropyAdj        the selection entropy adj
     * @param maxIterations              the max iterations
     * @param timeoutMinutes             the timeout minutes
     * @param seedPcaPower               the seed pca power
     * @param seedMagnitude              the seed magnitude
     */
    public VGG19(final int clusters, final int orientation, final double globalDistributionEmphasis, final double selectionEntropyAdj, final int maxIterations, final int timeoutMinutes, final double seedPcaPower, final double seedMagnitude) {
      super(clusters, orientation, globalDistributionEmphasis, selectionEntropyAdj, maxIterations, timeoutMinutes, seedPcaPower, seedMagnitude);
    }
    
    @Override
    public CVPipe_VGG19 getInstance() {
      return CVPipe_VGG19.INSTANCE;
    }
    
    @Override
    @Nonnull
    public CVPipe_VGG19.Layer[] getLayerTypes() {
      return CVPipe_VGG19.Layer.values();
    }
    
    
  }
  
}
