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

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.MutableResult;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandSelectLayer;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.layers.java.ImgPixelSumLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
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
public abstract class ImageSegmentation<T extends LayerEnum<T>, U extends CVPipe<T>> {
  
  private static final Logger logger = LoggerFactory.getLogger(ImageSegmentation.class);
  
  public static Tensor concat(final List<Tensor> featureImage) {
    assert featureImage.stream().map(Tensor::getDimensions).allMatch(dims -> dims.length == 2 || (dims.length == 3 && dims[2] == 1));
    int[] dimensions = featureImage.get(0).getDimensions();
    Tensor image = new Tensor(dimensions[0], dimensions[1], featureImage.size());
    image.setByCoord(c -> {
      int[] coords = c.getCoords();
      return featureImage.get(coords[2]).get(coords[0], coords[1]);
    });
    return image;
  }
  
  @Nonnull
  public static Tensor getDeltaMaskFromFeatureNetwork(final int[] imgDimensions, final Result featureResult, final Layer metricFilter) {
    DeltaSet<Layer> deltaBuffer = new DeltaSet<>();
    PipelineNetwork.build(1, unitImageNetwork(), metricFilter.copy().freeze(), new SumReducerLayer()).eval(featureResult).accumulate(deltaBuffer);
    if (deltaBuffer.getMap().size() != 1) throw new AssertionError(deltaBuffer.getMap().size());
    double[] delta = deltaBuffer.getMap().values().iterator().next().getDelta();
    return new Tensor(delta, imgDimensions[0], imgDimensions[1], 3);
  }
  
  @Nonnull
  public static PipelineNetwork unitImageNetwork() {
    PipelineNetwork pipelineNetwork = PipelineNetwork.build(1, new SquareActivationLayer(), new ImgPixelSumLayer(), new NthPowerActivationLayer().setPower(-0.5));
    pipelineNetwork.add(new ProductLayer(), pipelineNetwork.getInput(0), pipelineNetwork.getHead());
    return pipelineNetwork;
  }
  
  public static void display(@Nonnull final NotebookOutput log, final Tensor img, Tensor maskData) {
    if (maskData.getDimensions()[2] == 3) {
      display(log, img, PCAObjectLocation.sum3channels(maskData));
    }
    else {
      log.p(
        log.image(img.toRgbImageAlphaMask(0, 1, 2, maskData.rescaleRms(255.0)), "")
          + log.image(img.toRgbImageAlphaMask(0, 1, 2, maskData.normalizeDistribution().rescaleRms(255.0)), "")
      );
    }
  }
  
  /**
   * Run.
   *
   * @param log the log
   */
  public void run(@Nonnull final NotebookOutput log, Tensor img) {
    run(log, img, new MaxentImgClusterer(
        10,
        5,
        1,
        1,
        1,
        100,
        10,
        0
      ),
      CVPipe_VGG19.Layer.Layer_0,
      CVPipe_VGG19.Layer.Layer_1a,
//      CVPipe_VGG19.Layer.Layer_1b,
//      CVPipe_VGG19.Layer.Layer_1c,
      CVPipe_VGG19.Layer.Layer_1d
//      ,CVPipe_VGG19.Layer.Layer_1e
    );
  }
  
  public void run(@Nonnull final NotebookOutput log, final Tensor img, final MaxentImgClusterer clusterer, final CVPipe_VGG19.Layer... layers) {
    log.p(log.image(img.toImage(), ""));
    log.code(() -> {
      return clusterer;
    });
    List<Tensor> featureMasks = Arrays.stream(getLayerTypes()).filter(x -> Arrays.asList(layers).contains(x)).flatMap(layer -> {
      log.h2(layer.name());
      Layer network = getInstance().getPrototypes().get(layer);
      ArtistryUtil.setPrecision((DAGNetwork) network, Precision.Float);
      network.setFrozen(true);
      Result imageFeatures = network.eval(new MutableResult(img));
      Tensor featureImage = imageFeatures.getData().get(0);
      log.p("Feature Image Dimension: " + Arrays.toString(featureImage.getDimensions()));
      Layer maskNetwork = clusterer.analyze(featureImage, log);
      return display(log, img, clusterer, imageFeatures, maskNetwork).stream();
    }).collect(Collectors.toList());
    
    
    log.h2("Spatial Clusters");
    Tensor concat = concat(featureMasks.stream().map(x -> PCAObjectLocation.sum3channels(x)).collect(Collectors.toList()));
    display(log, img, clusterer, new MutableResult(concat), clusterer.analyze(concat, log));
    
  }
  
  public List<Tensor> display(@Nonnull final NotebookOutput log, final Tensor img, final MaxentImgClusterer clusterer, final Result imageFeatures, final Layer maskNetwork) {
    return IntStream.range(0, clusterer.getClusters()).mapToObj(i -> {
      Tensor maskData = getDeltaMaskFromFeatureNetwork(img.getDimensions(), imageFeatures, PipelineNetwork.build(1,
        maskNetwork,
        new ImgBandSelectLayer(i, i + 1)
      )).map(v -> Math.abs(v));
      display(log, img, maskData);
      return maskData;
    }).collect(Collectors.toList());
  }
  
  public abstract U getInstance();
  
  @Nonnull
  public abstract T[] getLayerTypes();
  
  /**
   * The type Vgg 19.
   */
  public static class VGG19 extends ImageSegmentation<CVPipe_VGG19.Layer, CVPipe_VGG19> {
    
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
