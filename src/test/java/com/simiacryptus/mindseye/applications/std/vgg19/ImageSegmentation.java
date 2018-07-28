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

package com.simiacryptus.mindseye.applications.std.vgg19;

import com.simiacryptus.mindseye.applications.ArtistryAppBase_VGG19;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.ImageSegmenter;
import com.simiacryptus.mindseye.applications.PCAObjectLocation;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The type Image classifier apply base.
 */
public class ImageSegmentation extends ArtistryAppBase_VGG19 {
  
  private static final Logger logger = LoggerFactory.getLogger(ImageSegmentation.class);
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    for (final Tensor img : loadImages_library()) {
      log.p(log.image(img.toImage(), ""));
      ImageSegmenter<CVPipe_VGG19.Layer, CVPipe_VGG19> segmenter = log.code(() -> {
        return new ImageSegmenter.VGG19(9) {
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
      });
      List<Tensor> featureMasks = segmenter.featureClusters(log, img,
        CVPipe_VGG19.Layer.Layer_0,
        CVPipe_VGG19.Layer.Layer_1a,
        CVPipe_VGG19.Layer.Layer_1e
      );
      for (int blur : Arrays.asList(9)) {
        for (int clusters : Arrays.asList(3)) {
          log.h2(String.format("Blur=%s, Clusters=%s", blur, clusters));
          segmenter.setClusters(clusters);
          segmenter.spatialClusters(log, img, PCAObjectLocation.blur(featureMasks, blur));
        }
      }
    }
  }
  
  /**
   * Load images 1 tensor [ ] [ ].
   *
   * @return the tensor [ ] [ ]
   */
  public Tensor[] loadImages_library() {
    List<CharSequence> localFiles = ArtistryUtil.getLocalFiles("H:\\SimiaCryptus\\Artistry\\Owned\\");
    Collections.shuffle(localFiles);
    //Stream<String> files = Stream.of("H:\\SimiaCryptus\\Artistry\\cat-and-dog.jpg");
    return localFiles.stream().map((CharSequence x1) -> x1.toString()).map(img -> {
      try {
        BufferedImage image = ImageIO.read(new File(img));
        image = TestUtil.resizePx(image, (long) 700 * 500);
        return Tensor.fromRGB(image);
      } catch (Throwable e) {
        return null;
      }
    }).filter(x -> x != null).toArray(i -> new Tensor[i]);
  }
  
  
}
