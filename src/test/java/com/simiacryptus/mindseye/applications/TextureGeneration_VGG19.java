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
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * The type Style transfer vgg 19.
 */
public class TextureGeneration_VGG19 extends ArtistryAppBase {
  
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
    init(log);
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
    double growthFactor = Math.sqrt(1.5);
    int trainingMinutes = 90;
    int phases = 2;
  
    BufferedImage canvas = initCanvas(new AtomicInteger(256));
    {
      double coeff_mean = 1e0;
      double coeff_cov = 1e0;
      Map<List<CharSequence>, TextureGeneration.StyleCoefficients> styles = new HashMap<>();
      styles.put(picasso, new TextureGeneration.StyleCoefficients(TextureGeneration.CenteringMode.Origin)
          .set(CVPipe_VGG19.Layer.Layer_1a, coeff_mean, coeff_cov)
          .set(CVPipe_VGG19.Layer.Layer_1b, coeff_mean, coeff_cov)
//        .set(CVPipe_VGG19.Layer.Layer_1d, coeff_mean, coeff_cov)
      );
      canvas = run(log, styleTransfer, precision, new AtomicInteger(256), growthFactor, styles, trainingMinutes, canvas, phases);
    }
  
    double coeff_mean = 1e1;
    double coeff_cov = 1e0;
    Map<List<CharSequence>, TextureGeneration.StyleCoefficients> styles = new HashMap<>();
    styles.put(picasso, new TextureGeneration.StyleCoefficients(TextureGeneration.CenteringMode.Origin)
      .set(CVPipe_VGG19.Layer.Layer_1c, coeff_mean, coeff_cov)
      .set(CVPipe_VGG19.Layer.Layer_1d, coeff_mean, coeff_cov)
    );
    canvas = run(log, styleTransfer, precision, new AtomicInteger(256), growthFactor, styles, trainingMinutes, canvas, phases);
  
    log.setFrontMatterProperty("status", "OK");
  }
  
  public BufferedImage run(@Nonnull final NotebookOutput log, final TextureGeneration.VGG19 styleTransfer, final Precision precision, final AtomicInteger imageSize, final double growthFactor, final Map<List<CharSequence>, TextureGeneration.StyleCoefficients> styles, final int trainingMinutes, BufferedImage canvasImage, final int phases) {
    log.h1("Phase 0");
    Map<CharSequence, BufferedImage> styleImages = new HashMap<>();
    TextureGeneration.StyleSetup styleSetup;
    TextureGeneration.NeuralSetup measureStyle;
    
    styleImages.clear();
    styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(x -> x, image -> ArtistryUtil.load(image, imageSize.get()))));
    styleSetup = new TextureGeneration.StyleSetup(precision, styleImages, styles);
    measureStyle = styleTransfer.measureStyle(styleSetup);
    
    canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
    canvasImage = styleTransfer.generate(server, log, canvasImage, styleSetup, trainingMinutes, measureStyle);
    for (int i = 1; i < phases; i++) {
      log.h1("Phase " + i);
      imageSize.set((int) (imageSize.get() * growthFactor));
      
      styleImages.clear();
      styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(x -> x, image -> ArtistryUtil.load(image, imageSize.get()))));
      styleSetup = new TextureGeneration.StyleSetup(precision, styleImages, styles);
      measureStyle = styleTransfer.measureStyle(styleSetup);
      
      canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
      canvasImage = styleTransfer.generate(server, log, canvasImage, styleSetup, trainingMinutes, measureStyle);
    }
    return canvasImage;
  }
  
  @Nonnull
  public BufferedImage initCanvas(final AtomicInteger imageSize) {
    BufferedImage canvasImage = ArtistryUtil.load(monkey, imageSize.get());
    canvasImage = ArtistryUtil.randomize(canvasImage, x -> 100 + 200 * (FastRandom.INSTANCE.random() - 0.5));
    canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
    canvasImage = ArtistryUtil.paint_Plasma(imageSize.get(), 3, 100.0, 1.4).toImage();
    return canvasImage;
  }
  
}
