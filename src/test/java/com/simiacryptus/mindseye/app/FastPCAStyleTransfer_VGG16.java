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
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

public class FastPCAStyleTransfer_VGG16 extends FastPCAStyleTransferBase<MultiLayerVGG16.LayerType, MultiLayerVGG16> {
  
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
    double growthFactor = Math.sqrt(1.5);
    String lakeAndForest = "H:\\SimiaCryptus\\Artistry\\Owned\\IMG_20170624_153541213-EFFECTS.jpg";
    String vanGogh = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
    String threeMusicians = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
    
    Map<String, StyleCoefficients> styles = new HashMap<>();
    double contentCoeff = 1e0;
    styles.put(lakeAndForest, new StyleCoefficients(true)
      .set(MultiLayerVGG16.LayerType.Layer_0, contentCoeff * 1e-5, contentCoeff * 1e-5)
    );
    styles.put(threeMusicians, new StyleCoefficients(false)
      .set(MultiLayerVGG16.LayerType.Layer_1d, 1e-5, 1e-5)
      .set(MultiLayerVGG16.LayerType.Layer_1e, 1e-5, 1e-5)
    );
    ContentCoefficients contentCoefficients = new ContentCoefficients()
//      .set(MultiLayerVGG16.LayerType.Layer_0, contentCoeff * 1e-6)
      .set(MultiLayerVGG16.LayerType.Layer_1b, contentCoeff * 1e-3);
    double power = 0.0;
    int trainingMinutes = 90;
    
    log.h1("Phase 0");
    BufferedImage canvasImage = load(lakeAndForest, imageSize);
    canvasImage = randomize(canvasImage);
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    Map<String, BufferedImage> styleImages = new HashMap<>();
    final int finalImageSize = imageSize;
    styles.forEach((file, parameters) -> styleImages.put(file, load(file, file == lakeAndForest ? ((int) (finalImageSize * 1.5)) : finalImageSize)));
    BufferedImage contentImage = load(lakeAndForest, canvasImage.getWidth(), canvasImage.getHeight());
    canvasImage = styleTransfer(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles, power), trainingMinutes);
    for (int i = 1; i < 10; i++) {
      log.h1("Phase " + i);
      imageSize = (int) (imageSize * growthFactor);
      styleImages.clear();
      final int finalImageSize1 = imageSize;
      styles.forEach((file, parameters) -> styleImages.put(file, load(file, file == lakeAndForest ? ((int) (finalImageSize1 * 1.5)) : finalImageSize1)));
      canvasImage = TestUtil.resize(canvasImage, imageSize, true);
      contentImage = load(lakeAndForest, canvasImage.getWidth(), canvasImage.getHeight());
      canvasImage = styleTransfer(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles, power), trainingMinutes);
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
