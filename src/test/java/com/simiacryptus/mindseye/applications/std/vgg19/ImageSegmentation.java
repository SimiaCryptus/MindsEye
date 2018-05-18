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
import com.simiacryptus.mindseye.applications.ImageSegmenter;
import com.simiacryptus.mindseye.applications.PCAObjectLocation;
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
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

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
      ImageSegmenter self = log.code(() -> {
        return new ImageSegmenter.VGG19(8);
      });
      List<Tensor> featureMasks = self.featureClusters(log, img,
        CVPipe_VGG19.Layer.Layer_0,
        CVPipe_VGG19.Layer.Layer_1a,
        CVPipe_VGG19.Layer.Layer_1b
//        CVPipe_VGG19.Layer.Layer_1c,
//        CVPipe_VGG19.Layer.Layer_1d,
//        CVPipe_VGG19.Layer.Layer_1e
      );
      self.spatialClusters(log, img, featureMasks);
      for (int blur : Arrays.asList(7, 5, 3)) {
        for (int clusters : Arrays.asList(5, 3, 2)) {
          log.h3(String.format("Blur=%s, Clusters=%s", blur, clusters));
          self.setClusters(clusters);
          self.spatialClusters(log, img, PCAObjectLocation.blur(featureMasks, blur));
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
    return Stream.of(
      "H:\\SimiaCryptus\\Artistry\\cat-and-dog.jpg"
    ).map(img -> {
      try {
        BufferedImage image = ImageIO.read(new File(img));
        image = TestUtil.resize(image, 400, true);
        return Tensor.fromRGB(image);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }).toArray(i -> new Tensor[i]);
  }
  
  
}
