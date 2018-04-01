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

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.MutableResult;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.models.ImageClassifier;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public abstract class ObjectLocationBase extends ArtistryAppBase {
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
    
    ImageClassifier classifier = getClassifierNetwork();
    Layer classifyNetwork = classifier.getNetwork();
    
    ImageClassifier locator = getLocatorNetwork();
    Layer locatorNetwork = locator.getNetwork();
    ArtistryAppBase.setPrecision((DAGNetwork) classifyNetwork, Precision.Float);
    ArtistryAppBase.setPrecision((DAGNetwork) locatorNetwork, Precision.Float);
    
    
    Tensor[][] inputData = loadImages_library();
//    Tensor[][] inputData = loadImage_Caltech101(log);
    double alphaPower = 0.8;
    
    final AtomicInteger index = new AtomicInteger(0);
    Arrays.stream(inputData).limit(10).forEach(row -> {
      log.h3("Image " + index.getAndIncrement());
      final Tensor img = row[0];
      try {
        log.p(log.image(img.toImage(), ""));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      Result classifyResult = classifyNetwork.eval(new MutableResult(row));
      Result locationResult = locatorNetwork.eval(new MutableResult(row));
      Tensor classification = classifyResult.getData().get(0);
      List<CharSequence> categories = classifier.getCategories();
      int[] sortedIndices = IntStream.range(0, categories.size()).mapToObj(x -> x)
        .sorted(Comparator.comparing(i -> -classification.get(i))).mapToInt(x -> x).limit(10).toArray();
      logger.info(Arrays.stream(sortedIndices)
        .mapToObj(i -> String.format("%s: %s = %s%%", i, categories.get(i), classification.get(i) * 100))
        .reduce((a, b) -> a + "\n" + b)
        .orElse(""));
      Map<CharSequence, Tensor> vectors = new HashMap<>();
      List<CharSequence> predictionList = Arrays.stream(sortedIndices).mapToObj(categories::get).collect(Collectors.toList());
      Arrays.stream(sortedIndices).limit(10).forEach(category -> {
        try {
          CharSequence name = categories.get(category);
          log.h3(name);
          Tensor alphaTensor = renderAlpha(alphaPower, img, locationResult, classification, category);
          log.p(log.image(img.toRgbImageAlphaMask(0, 1, 2, alphaTensor), ""));
          vectors.put(name, alphaTensor.unit());
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      
      Tensor avgDetection = vectors.values().stream().reduce((a, b) -> a.add(b)).get().scale(1.0 / vectors.size());
      Array2DRowRealMatrix covarianceMatrix = new Array2DRowRealMatrix(predictionList.size(), predictionList.size());
      for (int x = 0; x < predictionList.size(); x++) {
        for (int y = 0; y < predictionList.size(); y++) {
          Tensor l = vectors.get(predictionList.get(x)).minus(avgDetection);
          Tensor r = vectors.get(predictionList.get(y)).minus(avgDetection);
          covarianceMatrix.setEntry(x, y, l.dot(r));
        }
      }
      @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(covarianceMatrix);
      
      
      for (int objectVector = 0; objectVector < 10; objectVector++) {
        log.h3("Eigenobject " + objectVector);
        double eigenvalue = decomposition.getRealEigenvalue(objectVector);
        RealVector eigenvector = decomposition.getEigenvector(objectVector);
        Tensor detectionRegion = IntStream.range(0, eigenvector.getDimension()).mapToObj(i -> vectors.get(predictionList.get(i)).scale(eigenvector.getEntry(i))).reduce((a, b) -> a.add(b)).get();
        detectionRegion = detectionRegion.scale(255.0 / detectionRegion.rms());
        CharSequence categorization = IntStream.range(0, eigenvector.getDimension()).mapToObj(i -> {
          CharSequence category = predictionList.get(i);
          double component = eigenvector.getEntry(i);
          return String.format("<li>%s = %.4f</li>", category, component);
        }).reduce((a, b) -> a + "" + b).get();
        try {
          log.p(String.format("Object Detected: <ol>%s</ol>", categorization));
          log.p("Object Eigenvalue: " + eigenvalue);
          log.p("Object Region: " + log.image(img.toRgbImageAlphaMask(0, 1, 2, detectionRegion), ""));
          log.p("Object Region Compliment: " + log.image(img.toRgbImageAlphaMask(0, 1, 2, detectionRegion.scale(-1)), ""));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }

//      final int[] orderedVectors = IntStream.range(0, 10).mapToObj(x -> x)
//        .sorted(Comparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
//      IntStream.range(0, orderedVectors.length)
//        .mapToObj(i -> {
//            //double realEigenvalue = decomposition.getRealEigenvalue(orderedVectors[i]);
//            return decomposition.getEigenvector(orderedVectors[i]).toArray();
//          }
//        ).toArray(i -> new double[i][]);
      
      log.p(String.format("<table><tr><th>Cosine Distance</th>%s</tr>%s</table>",
        Arrays.stream(sortedIndices).limit(10).mapToObj(col -> "<th>" + categories.get(col) + "</th>").reduce((a, b) -> a + b).get(),
        Arrays.stream(sortedIndices).limit(10).mapToObj(r -> {
          return String.format("<tr><td>%s</td>%s</tr>", categories.get(r), Arrays.stream(sortedIndices).limit(10).mapToObj(col -> {
            return String.format("<td>%.4f</td>", Math.acos(vectors.get(categories.get(r)).dot(vectors.get(categories.get(col)))));
          }).reduce((a, b) -> a + b).get());
        }).reduce((a, b) -> a + b).orElse("")));
    });
    
    log.setFrontMatterProperty("status", "OK");
  }
  
  public abstract ImageClassifier getLocatorNetwork();
  
  public abstract ImageClassifier getClassifierNetwork();
  
  /**
   * Render alpha tensor.
   *
   * @param alphaPower     the alpha power
   * @param img            the img
   * @param locationResult the location result
   * @param classification the classification
   * @param category       the category
   * @return the tensor
   */
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
      "H:\\SimiaCryptus\\Artistry\\cat-and-dog.jpg",
      "H:\\SimiaCryptus\\Artistry\\pexels-photo-327011.jpg",
      "H:\\SimiaCryptus\\Artistry\\Defense.gov_News_Photo_120318-M-MM918-006_-_U.S._Marine_Cpl._Kyle_Click_and_his_military_working_dog_Windy_an_improvised_explosive_device_detection_dog_search_the_perimeter_of_the_Safar.jpg",
      "H:\\SimiaCryptus\\Artistry\\india_indian_family_happy_motorcycle_asian_together_family_father-1053028.jpg"
//      "H:\\SimiaCryptus\\Artistry\\rodeo.jpg",
//      "H:\\SimiaCryptus\\Artistry\\family.jpg",
//      "H:\\SimiaCryptus\\Artistry\\monkeydog.jpg",
//      "H:\\SimiaCryptus\\Artistry\\safari.jpg",
//      "H:\\SimiaCryptus\\Artistry\\wild-animals-group.jpg",
//      "H:\\SimiaCryptus\\Artistry\\girl_dog_family.jpg",
//      "H:\\SimiaCryptus\\Artistry\\chimps\\chip.jpg"
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
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
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
