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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Supplier;
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
    int phases = 3;
    int geometricEnd = 4 * 4;
    int maxIterations = 10;
    int trainingMinutes = 90;
    int startImageSize = 256;
    Supplier<DoubleStream> contentCoeffStream = () -> DoubleStream.iterate(5e-1, x -> x * Math.pow(1e2, 1.0 / 4)).limit(4);
    Supplier<DoubleStream> dreamCoeffStream = () -> DoubleStream.iterate(1e-3, x -> x * Math.pow(1e1, 1.0 / 4)).limit(4);
  
    ArtistryData.space.stream().forEach(contentSource -> {
      ArtistryData.CLASSIC_STYLES.stream().map(x -> Arrays.asList(x)).collect(Collectors.toList()).forEach(styleSources -> {
        dreamCoeffStream.get().forEach(dreamCoeff -> {
          BufferedImage[] imgs = contentCoeffStream.get().mapToObj(contentMixingCoeff -> {
            return styleTransfer(log, styleTransfer, precision, new AtomicInteger(startImageSize), Math.pow(geometricEnd, 1.0 / (2 * phases)), contentSource, create(x ->
              {
                x.put(styleSources, new StyleTransferBase.StyleCoefficients(StyleTransferBase.CenteringMode.Origin)
                    .set(CVPipe_VGG19.Layer.Layer_1a, 1e0, 1e0, dreamCoeff)
                    .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0, dreamCoeff)
                    .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, dreamCoeff)
                  //.set(CVPipe_VGG19.Layer.Layer_1d, 1e0, 1e0, dreamCoeff)
                );
              }),
              new StyleTransferBase.ContentCoefficients()
                .set(CVPipe_VGG19.Layer.Layer_1c, contentMixingCoeff)
                .set(CVPipe_VGG19.Layer.Layer_1d, contentMixingCoeff),
              trainingMinutes, maxIterations, phases);
          }).toArray(i -> new BufferedImage[i]);
          log.p(TestUtil.animatedGif(log,
            //IntStream.range(1-imgs.length,imgs.length).map(x->Math.abs(x)).mapToObj(i->imgs[i]).toArray(i->new BufferedImage[i])
            imgs
          ));
        });
      });
    });
    log.setFrontMatterProperty("status", "OK");
  }
  
  public BufferedImage styleTransfer(@Nonnull final NotebookOutput log, final StyleTransferBase.VGG19 styleTransfer, final Precision precision, final AtomicInteger imageSize, final double growthFactor, final CharSequence contentSource, final Map<List<CharSequence>, StyleTransferBase.StyleCoefficients> styles, final StyleTransferBase.ContentCoefficients contentCoefficients, final int trainingMinutes, final int maxIterations, final int phases) {
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
    return canvasImage;
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
