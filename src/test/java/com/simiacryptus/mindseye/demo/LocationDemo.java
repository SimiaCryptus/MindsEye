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

package com.simiacryptus.mindseye.demo;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.MutableResult;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image classifier run base.
 */
public class LocationDemo extends ArtistryDemo {
  
  
  /**
   * The Texture netork.
   */
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void run() {
    run(this::run);
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
  
    @Nonnull String logName = "cuda_" + log.getName() + ".log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CudaSystem.addLog(new PrintStream(log.file(logName)));
    
    VGG16_HDF5 vgg16_hdf5;
    try {
      vgg16_hdf5 = new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase3b() {
          add(new BandReducerLayer()
            .setMode(getFinalPoolingMode()));
          //add(new SoftmaxActivationLayer());
        }
      }//.setSamples(5).setDensity(0.3)
        .setFinalPoolingMode(PoolingLayer.PoolingMode.Avg);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  
    Layer classifyNetwork = vgg16_hdf5.getNetwork();
    
    Tensor[][] inputData = loadImages1();
//    Tensor[][] inputData = log.code(() -> {
//      return Caltech101.trainingDataStream().sorted(getShuffleComparator()).map(labeledObj -> {
//        @Nullable BufferedImage img = labeledObj.data.get();
//        img = TestUtil.resize(img, 224);
//        return new Tensor[]{Tensor.fromRGB(img)};
//      }).limit(10).toArray(i1 -> new Tensor[i1][]);
//    });
    
    
    final AtomicInteger index = new AtomicInteger(0);
    Arrays.stream(inputData).forEach(row -> {
      log.h3("Image " + index.getAndIncrement());
      try {
        log.p(log.image(row[0].toImage(), ""));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      Result classifyResult = classifyNetwork.eval(new MutableResult(row));
      Tensor classification = classifyResult.getData().get(0);
      List<String> categories = vgg16_hdf5.getCategories();
      int[] sortedIndices = IntStream.range(0, categories.size()).mapToObj(x -> x)
        .sorted(Comparator.comparing(i -> -classification.get(i))).mapToInt(x -> x).limit(100).toArray();
      logger.info(Arrays.stream(sortedIndices)
        .mapToObj(i -> String.format("%s: %s = %s%%", i, categories.get(i), classification.get(i) * 100))
        .reduce((a, b) -> a + "\n" + b)
        .orElse(""));
      Arrays.stream(sortedIndices).forEach(category -> {
        Tensor oneHot = new Tensor(classification.getDimensions()).set(category, 1);
        TensorArray tensorArray = TensorArray.wrap(oneHot);
        DeltaSet<Layer> deltaSet = new DeltaSet<>();
        classifyResult.accumulate(deltaSet, tensorArray);
        Tensor delta = new Tensor(deltaSet.getMap().entrySet().stream().filter(x -> x.getValue().target == row[0].getData()).findAny().get().getValue().getDelta(), row[0].getDimensions());
        delta = delta.mapAndFree(x -> Math.abs(x));
        try {
          log.h3(categories.get(category));
          log.p(log.image(TestUtil.normalizeBands(delta).toImage(), ""));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    });
    
    
    log.setFrontMatterProperty("status", "OK");
  }
  
  /**
   * Load images 1 tensor [ ] [ ].
   *
   * @return the tensor [ ] [ ]
   */
  public Tensor[][] loadImages1() {
    return Stream.of(
      "H:\\SimiaCryptus\\Artistry\\wild-animals-group.jpg",
      "H:\\SimiaCryptus\\Artistry\\girl_dog_family.jpg",
      "H:\\SimiaCryptus\\Artistry\\chimps\\chip.jpg"
    ).map(img -> {
      try {
        BufferedImage image = ImageIO.read(new File(img));
        image = TestUtil.resize(image, 400, true);
        return new Tensor[]{Tensor.fromRGB(image)};
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }).toArray(i -> new Tensor[i][]);
  }
  
  /**
   * Gets feature vector.
   *
   * @param log the log
   * @return the feature vector
   */
  protected Tensor getFeatureVector(@Nonnull final NotebookOutput log) {
    return log.code(() -> {
      return new Tensor(1, 1, 4096).setAll(0.0).set(5, 1.0);
    });
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  @Nonnull
  protected Class<?> getTargetClass() {
    return VGG16.class;
  }
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Demos;
  }
  
  /**
   * Gets shuffle comparator.
   *
   * @param <T> the type parameter
   * @return the shuffle comparator
   */
  public <T> Comparator<T> getShuffleComparator() {
    final int seed = (int) ((System.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    return Comparator.comparingInt(a1 -> System.identityHashCode(a1) ^ seed);
  }
}
