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
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Image classifier apply base.
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
  
    VGG16_HDF5 classifier;
    try {
      classifier = new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase3b() {
          add(new SoftmaxActivationLayer()
            .setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE)
            .setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL));
          add(new BandReducerLayer().setMode(getFinalPoolingMode()));
        }
      }//.setSamples(5).setDensity(0.3)
        .setFinalPoolingMode(PoolingLayer.PoolingMode.Max);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    Layer classifyNetwork = classifier.getNetwork();
  
    VGG16_HDF5 locator;
    try {
      locator = new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase3b() {
          add(new BandReducerLayer().setMode(getFinalPoolingMode()));
        }
      }//.setSamples(5).setDensity(0.3)
        .setFinalPoolingMode(PoolingLayer.PoolingMode.Avg);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    Layer locatorNetwork = locator.getNetwork();
  
  
    Tensor[][] inputData = loadImages_library();
//    Tensor[][] inputData = loadImage_Caltech101(log);
    double alphaPower = 0.8;
    
    final AtomicInteger index = new AtomicInteger(0);
    Arrays.stream(inputData).limit(10).forEach(row -> {
      log.h3("Image " + index.getAndIncrement());
      Tensor img = row[0];
      try {
        log.p(log.image(img.toImage(), ""));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      Result classifyResult = classifyNetwork.eval(new MutableResult(row));
      Result locationResult = locatorNetwork.eval(new MutableResult(row));
      Tensor classification = classifyResult.getData().get(0);
      List<String> categories = classifier.getCategories();
      int[] sortedIndices = IntStream.range(0, categories.size()).mapToObj(x -> x)
        .sorted(Comparator.comparing(i -> -classification.get(i))).mapToInt(x -> x).limit(10).toArray();
      logger.info(Arrays.stream(sortedIndices)
        .mapToObj(i -> String.format("%s: %s = %s%%", i, categories.get(i), classification.get(i) * 100))
        .reduce((a, b) -> a + "\n" + b)
        .orElse(""));
      List<Tensor> priorMatches = new ArrayList<>();
      Map<String, Tensor> vectors = new HashMap<>();
      Arrays.stream(sortedIndices).limit(10).forEach(category -> {
        try {
          log.h3(categories.get(category));
          Tensor alphaTensor = renderAlpha(alphaPower, img, locationResult, classification, category);
          Tensor orthogonalTensor = alphaTensor;
          for (Tensor t : priorMatches) {
            orthogonalTensor = orthogonalTensor.minus(t.scale(orthogonalTensor.dot(t) / t.sumSq()));
          }
          orthogonalTensor = orthogonalTensor.scale(Math.sqrt(alphaTensor.sumSq() / orthogonalTensor.sumSq()));
          priorMatches.add(alphaTensor);
          log.p("Raw Interest Region: " + log.image(row[0].toRgbImageAlphaMask(0, 1, 2, alphaTensor), ""));
          log.p("Orthogonal Interest Region: " + log.image(row[0].toRgbImageAlphaMask(0, 1, 2, orthogonalTensor), ""));
          vectors.put(categories.get(category), alphaTensor.unit());
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
  
      log.p(String.format("<table><th>Cosine Distance</th>%s%s</table>",
        String.format("<tr>%s</tr>", Arrays.stream(sortedIndices).limit(10).mapToObj(col -> "<th>" + categories.get(col) + "</th>").reduce((a, b) -> a + b)),
        Arrays.stream(sortedIndices).limit(10).mapToObj(r -> {
          return String.format("<tr><td>%s</td>%s</tr>", categories.get(r), Arrays.stream(sortedIndices).limit(10).mapToObj(col -> {
            return String.format("<td>%.4f</td>", Math.acos(vectors.get(categories.get(r)).dot(vectors.get(categories.get(col)))));
          }).reduce((a, b) -> a + b));
        }).reduce((a, b) -> a + b).orElse("")));
    });
    
    log.setFrontMatterProperty("status", "OK");
  }
  
  public Tensor renderAlpha(final double alphaPower, final Tensor img, final Result locationResult, final Tensor classification, final int category) {
    TensorArray tensorArray = TensorArray.wrap(new Tensor(classification.getDimensions()).set(category, 1));
    DeltaSet<Layer> deltaSet = new DeltaSet<>();
    locationResult.accumulate(deltaSet, tensorArray);
    double[] rawDelta = deltaSet.getMap().entrySet().stream().filter(x -> x.getValue().target == img.getData()).findAny().get().getValue().getDelta();
    Tensor deltaColor = new Tensor(rawDelta, img.getDimensions()).mapAndFree(x -> Math.abs(x));
    Tensor delta1d = blur(reduce(deltaColor), 3);
    return TestUtil.normalizeBands(TestUtil.normalizeBands(delta1d, 1).mapAndFree(x -> Math.pow(x, alphaPower)));
  }
  
  /**
   * Reduce tensor.
   *
   * @param deltaColor the delta color
   * @return the tensor
   */
  @Nonnull
  public Tensor reduce(final Tensor deltaColor) {
    return new Tensor(deltaColor.getDimensions()[0], deltaColor.getDimensions()[1], 1).setByCoord(c -> {
      return deltaColor.get(c.getCoords()[0], c.getCoords()[1], 0) +
        deltaColor.get(c.getCoords()[0], c.getCoords()[1], 1) +
        deltaColor.get(c.getCoords()[0], c.getCoords()[1], 2);
    });
  }
  
  /**
   * Blur tensor.
   *
   * @param delta1d    the delta 1 d
   * @param iterations the iterations
   * @return the tensor
   */
  @Nonnull
  public Tensor blur(Tensor delta1d, final int iterations) {
    ConvolutionLayer blur = new ConvolutionLayer(3, 3, 1, 1);
    blur.getKernel().set(0, 1, 1.0);
    blur.getKernel().set(1, 1, 1.0);
    blur.getKernel().set(1, 0, 1.0);
    blur.getKernel().set(1, 2, 1.0);
    blur.getKernel().set(2, 1, 1.0);
    for (int i = 0; i < iterations; i++) {
      delta1d = blur.eval(delta1d).getDataAndFree().getAndFree(0);
    }
    return delta1d;
  }
  
  /**
   * Load image caltech 101 tensor [ ] [ ].
   *
   * @param log the log
   * @return the tensor [ ] [ ]
   */
  public Tensor[][] loadImage_Caltech101(@Nonnull final NotebookOutput log) {
    return log.code(() -> {
      return Caltech101.trainingDataStream().sorted(getShuffleComparator()).map(labeledObj -> {
        @Nullable BufferedImage img = labeledObj.data.get();
        img = TestUtil.resize(img, 224);
        return new Tensor[]{Tensor.fromRGB(img)};
      }).limit(10).toArray(i1 -> new Tensor[i1][]);
    });
  }
  
  /**
   * Load images 1 tensor [ ] [ ].
   *
   * @return the tensor [ ] [ ]
   */
  public Tensor[][] loadImages_library() {
    return Stream.of(
      "H:\\SimiaCryptus\\Artistry\\rodeo.jpg",
      "H:\\SimiaCryptus\\Artistry\\family.jpg",
      "H:\\SimiaCryptus\\Artistry\\monkeydog.jpg",
      "H:\\SimiaCryptus\\Artistry\\safari.jpg",
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
