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

import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.MultiLayerVGG16;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class StyleTransfer_VGG16 extends StyleTransferBase<MultiLayerVGG16.LayerType, MultiLayerVGG16> {
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected Class<?> getTargetClass() {
    return VGG16.class;
  }
  
  public MultiLayerVGG16 getInstance() {
    return MultiLayerVGG16.INSTANCE;
  }
  
  @Nonnull
  public MultiLayerVGG16.LayerType[] getLayerTypes() {
    return MultiLayerVGG16.LayerType.values();
//    return new MultiLayerVGG16.LayerType[]{
//      MultiLayerVGG16.LayerType.Layer_0,
//      MultiLayerVGG16.LayerType.Layer_1a,
//      MultiLayerVGG16.LayerType.Layer_1b,
//      MultiLayerVGG16.LayerType.Layer_1c,
//      MultiLayerVGG16.LayerType.Layer_1d,
//      MultiLayerVGG16.LayerType.Layer_1e,
//      MultiLayerVGG16.LayerType.Layer_2a,
//      MultiLayerVGG16.LayerType.Layer_2b
//    };
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    init(log);
    Precision precision = Precision.Float;
    int imageSize = 400;
    parallelLossFunctions = true;
    double growthFactor = Math.sqrt(1.5);
    CharSequence lakeAndForest = "H:\\SimiaCryptus\\Artistry\\Owned\\IMG_20170624_153541213-EFFECTS.jpg";
    String monkey = "H:\\SimiaCryptus\\Artistry\\capuchin-monkey-2759768_960_720.jpg";
    CharSequence vanGogh = "H:\\SimiaCryptus\\Artistry\\portraits\\vangogh\\Van_Gogh_-_Portrait_of_Pere_Tanguy_1887-8.jpg";
    CharSequence threeMusicians = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
    CharSequence maJolie = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\Ma_Jolie_Pablo_Picasso.jpg";
  
    Map<List<CharSequence>, StyleCoefficients> styles = new HashMap<>();
    double coeff_mean = 1e0;
    double coeff_cov = 1e0;
    styles.put(Arrays.asList(threeMusicians, maJolie), new StyleCoefficients(CenteringMode.Dynamic)
//      .set(MultiLayerVGG16.LayerType.Layer_0, 1e0, 1e0)
//        .set(MultiLayerVGG16.LayerType.Layer_1a, coeff_mean, coeff_cov)
        .set(MultiLayerVGG16.LayerType.Layer_1b, coeff_mean, coeff_cov)
        .set(MultiLayerVGG16.LayerType.Layer_1c, coeff_mean, coeff_cov)
//        .set(MultiLayerVGG16.LayerType.Layer_1d, coeff_mean, coeff_cov)
    );
//    styles.put(Arrays.asList(vanGogh), new StyleCoefficients(true)
////      .set(MultiLayerVGG16.LayerType.Layer_1a, 1e0, 1e0)
////      .set(MultiLayerVGG16.LayerType.Layer_1b, 1e0, 1e0)
////      .set(MultiLayerVGG16.LayerType.Layer_1c, 1e0, 1e0)
////      .set(MultiLayerVGG16.LayerType.Layer_1d, 1e0, 1e0)
//    );
    ContentCoefficients contentCoefficients = new ContentCoefficients()
//      .set(MultiLayerVGG16.LayerType.Layer_1c, 1e-2)
      ;
    int trainingMinutes = 90;
    
    log.h1("Phase 0");
    BufferedImage canvasImage = load(monkey, imageSize);
    canvasImage = TestUtil.resize(TestUtil.resize(canvasImage, 16, true), imageSize, true);
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
//    canvasImage = randomize(canvasImage, x -> 10 * (FastRandom.INSTANCE.random()) * (FastRandom.INSTANCE.random() < 0.9 ? 1 : 0));
    canvasImage = randomize(canvasImage, x -> 127 + 2 * 127 * (FastRandom.INSTANCE.random() - 0.5));
//    canvasImage = randomize(canvasImage, x -> 10*(FastRandom.INSTANCE.random()-0.5));
//    canvasImage = randomize(canvasImage, x -> x*(FastRandom.INSTANCE.random()));
    Map<CharSequence, BufferedImage> styleImages = new HashMap<>();
    styleImages.clear();
    styleImages.putAll(styles.keySet().stream().flatMap(x -> x.stream()).collect(Collectors.toMap(x -> x, file -> load(file))));
    StyleSetup styleSetup = new StyleSetup(precision,
      load(monkey, canvasImage.getWidth(), canvasImage.getHeight()),
      contentCoefficients, styleImages, styles);
    NeuralSetup measureStyle = measureStyle(styleSetup);
    canvasImage = styleTransfer(log, canvasImage, styleSetup, trainingMinutes, measureStyle);
    for (int i = 1; i < 10; i++) {
      log.h1("Phase " + i);
      imageSize = (int) (imageSize * growthFactor);
      canvasImage = TestUtil.resize(canvasImage, imageSize, true);
      styleSetup.contentImage = load(monkey, canvasImage.getWidth(), canvasImage.getHeight());
      canvasImage = styleTransfer(log, canvasImage, styleSetup, trainingMinutes, measureStyle);
    }
    
    log.setFrontMatterProperty("status", "OK");
  }
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void run() {
    run(this::run, "StyleTransfer_" + new SimpleDateFormat("yyyyMMddHHmm").format(new Date()));
  }
  
}
