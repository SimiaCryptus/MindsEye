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
import com.simiacryptus.mindseye.applications.ArtistryData;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.StyleTransferBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * The type Style transfer vgg 19.
 */
public class StyleTransfer extends ArtistryAppBase_VGG19 {
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    StyleTransferBase.VGG19 styleTransfer = new StyleTransferBase.VGG19();
    init(log);
    Precision precision = Precision.Float;
    final AtomicInteger imageSize = new AtomicInteger(256);
    styleTransfer.parallelLossFunctions = true;
    double growthFactor = Math.sqrt(4);
  
    Map<List<CharSequence>, StyleTransferBase.StyleCoefficients> styles = new HashMap<>();
    List<CharSequence> styleSources = ArtistryData.CLASSIC_STYLES;
    styles.put(styleSources, new StyleTransferBase.StyleCoefficients(StyleTransferBase.CenteringMode.Origin)
        .set(CVPipe_VGG19.Layer.Layer_1a, 1e0, 1e0)
        .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
        .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0)
    );
    StyleTransferBase.ContentCoefficients contentCoefficients = new StyleTransferBase.ContentCoefficients()
      .set(CVPipe_VGG19.Layer.Layer_1b, 3e0)
      .set(CVPipe_VGG19.Layer.Layer_1c, 3e0);
    int trainingMinutes = 90;
    int maxIterations = 10;
    int phases = 2;
    
    log.h1("Phase 0");
    BufferedImage canvasImage = ArtistryUtil.load(ArtistryData.CLASSIC_STYLES.get(0), imageSize.get());
    canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
    canvasImage = ArtistryUtil.expandPlasma(Tensor.fromRGB(TestUtil.resize(canvasImage, 16, true)), imageSize.get(), 1000.0, 1.1).toImage();
    BufferedImage contentImage = ArtistryUtil.load(ArtistryData.CLASSIC_CONTENT.get(0), canvasImage.getWidth(), canvasImage.getHeight());
    Map<CharSequence, BufferedImage> styleImages = new HashMap<>();
    StyleTransferBase.StyleSetup styleSetup;
    
    styleImages.clear();
    styleImages.putAll(styles.keySet().stream().flatMap(x -> x.stream()).collect(Collectors.toMap(x -> x, file -> ArtistryUtil.load(file, imageSize.get()))));
    styleSetup = new StyleTransferBase.StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles);
  
    StyleTransferBase.NeuralSetup measureStyle = styleTransfer.measureStyle(styleSetup);
    canvasImage = styleTransfer.styleTransfer(log.getHttpd(), log, canvasImage, styleSetup, trainingMinutes, measureStyle, maxIterations);
    for (int i = 1; i < phases; i++) {
      log.h1("Phase " + i);
      imageSize.set((int) (imageSize.get() * growthFactor));
      canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
  
      styleImages.clear();
      styleImages.putAll(styles.keySet().stream().flatMap(x -> x.stream()).collect(Collectors.toMap(x -> x, file -> ArtistryUtil.load(file, imageSize.get()))));
      styleSetup = new StyleTransferBase.StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles);
    
      canvasImage = styleTransfer.styleTransfer(log.getHttpd(), log, canvasImage, styleSetup, trainingMinutes, measureStyle, maxIterations);
    }
    log.setFrontMatterProperty("status", "OK");
  }
  
}
