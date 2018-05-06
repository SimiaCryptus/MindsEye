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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
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
    com.simiacryptus.mindseye.applications.ImageSegmentation self = new com.simiacryptus.mindseye.applications.ImageSegmentation.VGG19();
    for (final Tensor img : loadImages_library()) {
      self.run(log, img);
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
