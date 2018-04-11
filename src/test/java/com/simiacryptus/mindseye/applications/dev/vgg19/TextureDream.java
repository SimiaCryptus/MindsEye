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
import com.simiacryptus.mindseye.applications.DeepDreamBase;
import com.simiacryptus.mindseye.applications.TextureGenerationBase;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.util.FileNanoHTTPD;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * The type Style transfer vgg 19.
 */
public class TextureDream extends ArtistryAppBase_VGG19 {
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    TextureGenerationBase.VGG19 styleTransfer = new TextureGenerationBase.VGG19();
    DeepDreamBase.VGG19 deepDream = new DeepDreamBase.VGG19();
    deepDream.setTiled(true);
    init(log);
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
  
    double growthFactor = Math.sqrt(2);
    int imageSize = 256;
    int styleSize = 512;
    int phases = 1;
    int maxIterations = 10;
    int trainingMinutes = 90;

    Arrays.asList(
      Arrays.asList(threeMusicians),
      waldo.subList(0,1),
      Arrays.asList(maJolie)
    ).forEach(styleSources->{
  
      final FileNanoHTTPD server = log.getHttpd();
      TextureGenerationBase.generate(log, styleTransfer, precision, imageSize, growthFactor, create(map ->
        map.put(styleSources, new TextureGenerationBase.StyleCoefficients(TextureGenerationBase.CenteringMode.Origin)
          .set(CVPipe_VGG19.Layer.Layer_1a, 1e0, 1e0)
          .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 1e0)
        )), trainingMinutes, TextureGenerationBase.initCanvas(new AtomicInteger(imageSize)), phases, maxIterations, server, styleSize);
  
      TextureGenerationBase.generate(log, styleTransfer, precision, imageSize, growthFactor, create(map ->
        map.put(styleSources, new TextureGenerationBase.StyleCoefficients(TextureGenerationBase.CenteringMode.Origin)
          .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
          .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 1e0)
        )), trainingMinutes, TextureGenerationBase.initCanvas(new AtomicInteger(imageSize)), phases, maxIterations, server, styleSize);
  
      TextureGenerationBase.generate(log, styleTransfer, precision, imageSize, growthFactor, create(map ->
        map.put(styleSources, new TextureGenerationBase.StyleCoefficients(TextureGenerationBase.CenteringMode.Origin)
          .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0, 1e0)
          .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0)
        )), trainingMinutes, TextureGenerationBase.initCanvas(new AtomicInteger(imageSize)), phases, maxIterations, server, styleSize);
  
      TextureGenerationBase.generate(log, styleTransfer, precision, imageSize, growthFactor, create(map ->
        map.put(styleSources, new TextureGenerationBase.StyleCoefficients(TextureGenerationBase.CenteringMode.Origin)
          .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
          .set(CVPipe_VGG19.Layer.Layer_1d, 1e0, 1e0, 1e0)
        )), trainingMinutes, TextureGenerationBase.initCanvas(new AtomicInteger(imageSize)), phases, maxIterations, server, styleSize);
  
      TextureGenerationBase.generate(log, styleTransfer, precision, imageSize, growthFactor, create(map ->
        map.put(styleSources, new TextureGenerationBase.StyleCoefficients(TextureGenerationBase.CenteringMode.Origin)
          .set(CVPipe_VGG19.Layer.Layer_1a, 1e0, 1e0)
          .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 0)
        )), trainingMinutes, TextureGenerationBase.initCanvas(new AtomicInteger(imageSize)), phases, maxIterations, server, styleSize);
  
      TextureGenerationBase.generate(log, styleTransfer, precision, imageSize, growthFactor, create(map ->
        map.put(styleSources, new TextureGenerationBase.StyleCoefficients(TextureGenerationBase.CenteringMode.Origin)
          .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
          .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 0)
        )), trainingMinutes, TextureGenerationBase.initCanvas(new AtomicInteger(imageSize)), phases, maxIterations, server, styleSize);
  
      TextureGenerationBase.generate(log, styleTransfer, precision, imageSize, growthFactor, create(map ->
        map.put(styleSources, new TextureGenerationBase.StyleCoefficients(TextureGenerationBase.CenteringMode.Origin)
          .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0, 0)
          .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0)
        )), trainingMinutes, TextureGenerationBase.initCanvas(new AtomicInteger(imageSize)), phases, maxIterations, server, styleSize);
  
      TextureGenerationBase.generate(log, styleTransfer, precision, imageSize, growthFactor, create(map ->
        map.put(styleSources, new TextureGenerationBase.StyleCoefficients(TextureGenerationBase.CenteringMode.Origin)
          .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
          .set(CVPipe_VGG19.Layer.Layer_1d, 1e0, 1e0, 0)
        )), trainingMinutes, TextureGenerationBase.initCanvas(new AtomicInteger(imageSize)), phases, maxIterations, server, styleSize);
    });

    log.setFrontMatterProperty("status", "OK");
  }
  
  @Nonnull
  public <K,V> Map<K, V> create(Consumer<Map<K, V>> configure) {
    Map<K, V> map = new HashMap<>();
    configure.accept(map);
    return map;
  }
  
}
