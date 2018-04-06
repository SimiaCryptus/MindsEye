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
public class StyleTransfer_VGG19 extends ArtistryAppBase_VGG19 {
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    StyleTransfer.VGG19 styleTransfer = new StyleTransfer.VGG19();
    init(log);
    Precision precision = Precision.Float;
    final AtomicInteger imageSize = new AtomicInteger(256);
    styleTransfer.parallelLossFunctions = true;
    double growthFactor = Math.sqrt(1.5);
    
    Map<List<CharSequence>, StyleTransfer.StyleCoefficients> styles = new HashMap<>();
    List<CharSequence> styleSources = vangogh;
    styles.put(styleSources, new StyleTransfer.StyleCoefficients(StyleTransfer.CenteringMode.Origin)
//      .set(CVPipe_VGG19.Layer.Layer_0, 1e0, 1e0)
        .set(CVPipe_VGG19.Layer.Layer_1a, 1e0, 1e0)
        .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
        .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0)
        .set(CVPipe_VGG19.Layer.Layer_1d, 1e0, 1e0)
    );
    StyleTransfer.ContentCoefficients contentCoefficients = new StyleTransfer.ContentCoefficients()
      .set(CVPipe_VGG19.Layer.Layer_1b, 3e0)
      .set(CVPipe_VGG19.Layer.Layer_1c, 3e0);
    int trainingMinutes = 90;
    int maxIterations = 150;
    
    log.h1("Phase 0");
    BufferedImage canvasImage = ArtistryUtil.load(monkey, imageSize.get());
    canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
    canvasImage = ArtistryUtil.expandPlasma(Tensor.fromRGB(TestUtil.resize(canvasImage, 16, true)), imageSize.get(), 1000.0, 1.1).toImage();
    BufferedImage contentImage = ArtistryUtil.load(monkey, canvasImage.getWidth(), canvasImage.getHeight());
    Map<CharSequence, BufferedImage> styleImages = new HashMap<>();
    StyleTransfer.StyleSetup styleSetup;
    
    styleImages.clear();
    styleImages.putAll(styles.keySet().stream().flatMap(x -> x.stream()).collect(Collectors.toMap(x -> x, file -> ArtistryUtil.load(file, imageSize.get()))));
    styleSetup = new StyleTransfer.StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles);
    
    StyleTransfer.NeuralSetup measureStyle = styleTransfer.measureStyle(styleSetup);
    canvasImage = styleTransfer.styleTransfer(server, log, canvasImage, styleSetup, trainingMinutes, measureStyle, maxIterations);
    for (int i = 1; i < 3; i++) {
      log.h1("Phase " + i);
      imageSize.set((int) (imageSize.get() * growthFactor));
      canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
  
      styleImages.clear();
      styleImages.putAll(styles.keySet().stream().flatMap(x -> x.stream()).collect(Collectors.toMap(x -> x, file -> ArtistryUtil.load(file, imageSize.get()))));
      styleSetup = new StyleTransfer.StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles);
  
      canvasImage = styleTransfer.styleTransfer(server, log, canvasImage, styleSetup, trainingMinutes, measureStyle, maxIterations);
    }
    log.setFrontMatterProperty("status", "OK");
  }
  
}
