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
import com.simiacryptus.mindseye.models.MultiLayerVGG19;
import com.simiacryptus.mindseye.models.VGG19;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;

public class DeepDream_VGG19 extends DeepDreamBase<MultiLayerVGG19.LayerType, MultiLayerVGG19> {
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected Class<?> getTargetClass() {
    return VGG19.class;
  }
  
  public MultiLayerVGG19 getInstance() {
    return MultiLayerVGG19.INSTANCE;
  }
  
  @Nonnull
  public MultiLayerVGG19.LayerType[] getLayerTypes() {
    return MultiLayerVGG19.LayerType.values();
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    init(log);
    Precision precision = Precision.Float;
    int imageSize = 800;
    String lakeAndForest = "H:\\SimiaCryptus\\Artistry\\Owned\\IMG_20170624_153541213-EFFECTS.jpg";
    CharSequence vanGogh = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
    CharSequence threeMusicians = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
  
    Map<MultiLayerVGG19.LayerType, ContentCoefficients> contentCoefficients = new HashMap<>();
    contentCoefficients.put(MultiLayerVGG19.LayerType.Layer_1d, new ContentCoefficients(0, 1e-1));
    contentCoefficients.put(MultiLayerVGG19.LayerType.Layer_1e, new ContentCoefficients(0, 1e0));
    contentCoefficients.put(MultiLayerVGG19.LayerType.Layer_2b, new ContentCoefficients(0, 1e1));
    int trainingMinutes = 180;
    
    log.h1("Phase 0");
    BufferedImage canvasImage = load(lakeAndForest, imageSize);
    //canvasImage = randomize(canvasImage);
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    BufferedImage contentImage = load(lakeAndForest, canvasImage.getWidth(), canvasImage.getHeight());
    deepDream(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients), trainingMinutes);
    
    log.setFrontMatterProperty("status", "OK");
  }
  
}
