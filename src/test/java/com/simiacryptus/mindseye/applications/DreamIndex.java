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

import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.VGG19;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Style transfer vgg 19.
 */
public class DreamIndex extends ArtistryAppBase {
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected Class<?> getTargetClass() {
    return VGG19.class;
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    TextureGeneration.VGG19 styleTransfer = new TextureGeneration.VGG19();
    DeepDream.VGG19 deepDream = new DeepDream.VGG19();
    deepDream.setTiled(true);
    init(log);
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
    double growthFactor = Math.sqrt(2);
    int iterations = 10;
    int trainingMinutes = 90;
    
    for (CharSequence file : picasso) {
      log.h2("Image: " + file);
      try {
        log.p(log.image(ImageIO.read(new File(file.toString())), "Input Image"));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      for (final CVPipe_VGG19.Layer layer : CVPipe_VGG19.Layer.values()) {
        BufferedImage canvas = TextureGeneration.initCanvas(new AtomicInteger(256));
        log.h2("Layer: " + layer);
        Map<List<CharSequence>, TextureGeneration.StyleCoefficients> textureStyle = new HashMap<>();
        textureStyle.put(Arrays.asList(file), new TextureGeneration.StyleCoefficients(TextureGeneration.CenteringMode.Origin)
          .set(layer, 1e0, 1e0));
        canvas = TextureGeneration.generate(log, styleTransfer, precision, new AtomicInteger(256), growthFactor, textureStyle, trainingMinutes, canvas, 1, iterations, server);
        Map<CVPipe_VGG19.Layer, DeepDream.ContentCoefficients> dreamCoeff = new HashMap<>();
        dreamCoeff.put(layer, new DeepDream.ContentCoefficients(0, 1e0));
        canvas = deepDream.deepDream(server, log, canvas, new DeepDream.StyleSetup(precision, canvas, dreamCoeff), trainingMinutes, iterations);
      }
    }
    
    log.setFrontMatterProperty("status", "OK");
  }
  
}
