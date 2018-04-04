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

package com.simiacryptus.mindseye.app;

import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.TextureGeneration;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.VGG19;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
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
    final AtomicInteger imageSize = new AtomicInteger(256);
    styleTransfer.parallelLossFunctions = true;
    double growthFactor = Math.sqrt(1.5);
    CharSequence lakeAndForest = "H:\\SimiaCryptus\\Artistry\\Owned\\IMG_20170624_153541213-EFFECTS.jpg";
    String monkey = "H:\\SimiaCryptus\\Artistry\\capuchin-monkey-2759768_960_720.jpg";
    CharSequence vanGogh1 = "H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\Van_Gogh_-_Portrait_of_Pere_Tanguy_1887-8.jpg";
    CharSequence vanGogh2 = "H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\800px-Vincent_van_Gogh_-_Dr_Paul_Gachet_-_Google_Art_Project.jpg";
    CharSequence threeMusicians = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
    CharSequence maJolie = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\Ma_Jolie_Pablo_Picasso.jpg";
    
    Map<List<CharSequence>, TextureGeneration.StyleCoefficients> styles = new HashMap<>();
    double coeff_mean = 1e0;
    double coeff_cov = 1e0;
    styles.put(Arrays.asList(
      //threeMusicians, maJolie
      vanGogh1, vanGogh2
      ), new TextureGeneration.StyleCoefficients(TextureGeneration.CenteringMode.Origin)
//      .set(CVPipe_VGG19.Layer.Layer_0, 1e0, 1e0)
//        .set(CVPipe_VGG19.Layer.Layer_1a, coeff_mean, coeff_cov)
        .set(CVPipe_VGG19.Layer.Layer_1b, coeff_mean, coeff_cov)
//        .set(CVPipe_VGG19.Layer.Layer_1c, coeff_mean, coeff_cov)
        .set(CVPipe_VGG19.Layer.Layer_1d, coeff_mean, coeff_cov)
    );
    int trainingMinutes = 90;
    
    log.h1("Phase 0");
    BufferedImage canvasImage = ArtistryUtil.load(monkey, imageSize.get());
    canvasImage = ArtistryUtil.randomize(canvasImage, x -> 100 + 200 * (FastRandom.INSTANCE.random() - 0.5));
    canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
    canvasImage = ArtistryUtil.paint_Plasma(imageSize.get(), 3, 100.0, 1.4).toImage();
    Map<CharSequence, BufferedImage> styleImages = new HashMap<>();
    TextureGeneration.StyleSetup styleSetup;
    
    styleImages.clear();
    styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(x -> x, image -> ArtistryUtil.load(image, imageSize.get()))));
    styleSetup = new TextureGeneration.StyleSetup(precision, styleImages, styles);
    
    TextureGeneration.NeuralSetup measureStyle = styleTransfer.measureStyle(styleSetup);
    canvasImage = styleTransfer.generate(server, log, canvasImage, styleSetup, trainingMinutes, measureStyle);
    for (int i = 1; i < 3; i++) {
      log.h1("Phase " + i);
      imageSize.set((int) (imageSize.get() * growthFactor));
      canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
      
      styleImages.clear();
      styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(x -> x, image -> ArtistryUtil.load(image, imageSize.get()))));
      styleSetup = new TextureGeneration.StyleSetup(precision, styleImages, styles);
      
      canvasImage = styleTransfer.generate(server, log, canvasImage, styleSetup, trainingMinutes, measureStyle);
    }
    log.setFrontMatterProperty("status", "OK");
  }
  
}
