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
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Object location.
 */
public abstract class ImageSegmenter<T extends LayerEnum<T>, U extends CVPipe<T>> extends PixelClusterer {
  
  private static final Logger logger = LoggerFactory.getLogger(ImageSegmenter.class);
  
  public ImageSegmenter(final int clusters, final int orientation, final double globalDistributionEmphasis, final double selectionEntropyAdj, final int maxIterations, final int timeoutMinutes, final double seedPcaPower, final double seedMagnitude) {
    super(clusters, orientation, globalDistributionEmphasis, selectionEntropyAdj, maxIterations, timeoutMinutes, seedPcaPower, seedMagnitude);
  }
  
  public ImageSegmenter(final int clusters) {
    super(clusters);
  }
  
  public static void displayImageMask(@Nonnull final NotebookOutput log, final Tensor img, Tensor mask) {
    log.p(log.image(img.toRgbImageAlphaMask(0, 1, 2, mask.scale(255.0)), "") +
      log.image(img.toRgbImageAlphaMask(0, 1, 2, mask.normalizeDistribution().scale(255.0)), ""));
  }
  
  public List<Tensor> featureClusters(@Nonnull final NotebookOutput log, final Tensor img, final T... layers) {
    return Arrays.stream(getLayerTypes()).filter(x -> Arrays.asList(layers).contains(x)).flatMap(layer -> {
      log.h2(layer.name());
      Layer network = getInstance().getPrototypes().get(layer);
      ArtistryUtil.setPrecision((DAGNetwork) network, Precision.Float);
      network.setFrozen(true);
      Result imageFeatures = network.eval(new MutableResult(img));
      Tensor featureImage = imageFeatures.getData().get(0);
      log.p("Feature Image Dimension: " + Arrays.toString(featureImage.getDimensions()));
      Layer analyze1 = analyze(log, featureImage);
      return IntStream.range(0, getClusters()).mapToObj(i -> {
        Tensor maskData = new Tensor(((Layer) PipelineNetwork.build(1,
          analyze1,
          new ImgBandSelectLayer(i, i + 1)
        )).copy().freeze().andThen(new SumReducerLayer()).eval(imageFeatures).getSingleDelta(), img.getDimensions()).map(v -> Math.abs(v));
        displayImageMask(log, img, maskData.sumChannels().rescaleRms(1.0));
        return maskData;
      }).collect(Collectors.toList()).stream();
    }).collect(Collectors.toList());
  }
  
  public List<Tensor> spatialClusters(@Nonnull final NotebookOutput log, final Tensor img, final List<Tensor> featureMasks) {
    log.h2("Spatial Clusters - " + getClusters());
    Tensor concat = ImgConcatLayer.eval(featureMasks.stream().map(Tensor::sumChannels).collect(Collectors.toList()));
    Tensor reclustered = analyze(log, concat).eval(concat).getDataAndFree().getAndFree(0);
    List<Tensor> tensorList = IntStream.range(0, reclustered.getDimensions()[2]).mapToObj(i -> reclustered.selectBand(i)).collect(Collectors.toList());
    for (Tensor selectBand : tensorList) {
      displayImageMask(log, img, selectBand);
    }
    return tensorList;
  }
  
  public abstract U getInstance();
  
  @Nonnull
  public abstract T[] getLayerTypes();
  
  /**
   * The type Vgg 19.
   */
  public static class VGG19 extends ImageSegmenter<CVPipe_VGG19.Layer, CVPipe_VGG19> {
    
    public VGG19(final int clusters) {
      super(clusters);
    }
    
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
