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

package com.simiacryptus.mindseye.applications.dev.vgg19;

import com.simiacryptus.mindseye.applications.ArtistryAppBase_VGG19;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.StyleTransferBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.test.TestUtil;
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
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * The type Style transfer vgg 19.
 */
public class ArtisticGradient extends ArtistryAppBase_VGG19 {
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    StyleTransferBase.VGG19 styleTransfer = new StyleTransferBase.VGG19();
    init(log);
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
    styleTransfer.setTiled(false);
    int phases = 4;
    int geometricEnd = 4 * 4;
    int maxIterations = 5;
    int trainingMinutes = 90;
    int startImageSize = 256;
    int sizeSteps = 6;
    double totalMag = 1e2;
    double minContentCoeff = 2e-1;
    Arrays.asList(
      vangogh.subList(0, 1),
      figures.subList(0, 1),
      picasso.subList(0, 1)
    ).forEach(styleSources -> {
      styleSources.forEach(img -> {
        try {
          log.p(log.image(ImageIO.read(new File(img.toString())), "Source Style Image"));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      owned.stream().limit(2).forEach(contentSource -> {
        DoubleStream.iterate(minContentCoeff, x -> x * Math.pow(totalMag, 1.0 / sizeSteps)).limit(sizeSteps).forEach(contentMixingCoeff -> {
          styleTransfer(log, styleTransfer, precision, new AtomicInteger(startImageSize), Math.pow(geometricEnd, 1.0 / (2 * phases)), contentSource, create(x ->
              x.put(styleSources, new StyleTransferBase.StyleCoefficients(StyleTransferBase.CenteringMode.Origin)
                .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0, 1e0)
                .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 1e0)
                .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 1e0)
              )),
            new StyleTransferBase.ContentCoefficients()
              .set(CVPipe_VGG19.Layer.Layer_1c, contentMixingCoeff)
              .set(CVPipe_VGG19.Layer.Layer_1d, contentMixingCoeff),
            trainingMinutes, maxIterations, phases);
        });
      });
    });
    log.setFrontMatterProperty("status", "OK");
  }
  
  public void styleTransfer(@Nonnull final NotebookOutput log, final StyleTransferBase.VGG19 styleTransfer, final Precision precision, final AtomicInteger imageSize, final double growthFactor, final CharSequence contentSource, final Map<List<CharSequence>, StyleTransferBase.StyleCoefficients> styles, final StyleTransferBase.ContentCoefficients contentCoefficients, final int trainingMinutes, final int maxIterations, final int phases) {
    BufferedImage canvasImage = init(contentSource, imageSize.get());
    for (int i = 0; i < phases; i++) {
      if (0 < i) {
        imageSize.set((int) (imageSize.get() * growthFactor));
        canvasImage = TestUtil.resize(canvasImage, imageSize.get(), true);
      }
      StyleTransferBase.StyleSetup styleSetup = new StyleTransferBase.StyleSetup(precision,
        ArtistryUtil.load(contentSource, canvasImage.getWidth(), canvasImage.getHeight()),
        contentCoefficients, create(y->y.putAll(styles.keySet().stream().flatMap(x -> x.stream())
        .collect(Collectors.toMap(x -> x, file -> ArtistryUtil.load(file, imageSize.get()))))), styles);
      canvasImage = styleTransfer.styleTransfer(log.getHttpd(), log, canvasImage, styleSetup,
        trainingMinutes, styleTransfer.measureStyle(styleSetup), maxIterations);
    }
  }
  
  @Nonnull
  public static BufferedImage init(final CharSequence contentSource, final int width) {
    BufferedImage canvasImage;
    canvasImage = ArtistryUtil.load(contentSource, width);
    canvasImage = TestUtil.resize(canvasImage, width, true);
    canvasImage = ArtistryUtil.expandPlasma(Tensor.fromRGB(
      TestUtil.resize(canvasImage, 16, true)),
      width, 1000.0, 1.1).toImage();
    return canvasImage;
  }
  
  @Nonnull
  public static <K, V> Map<K, V> create(Consumer<Map<K, V>> configure) {
    Map<K, V> map = new HashMap<>();
    configure.accept(map);
    return map;
  }
  
}
