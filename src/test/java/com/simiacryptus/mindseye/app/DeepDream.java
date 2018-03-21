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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.SquareActivationLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.lang.Tuple2;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * This notebook implements the Deep Dream protocol outlined in <a href="https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html">Inceptionism: Going Deeper into Neural Networks</a>
 */
public class DeepDream extends StyleTransfer {
  
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void run() {
    run(this::run, "DeepDream_" + new SimpleDateFormat("yyyyMMddHHmm").format(new Date()));
  }
  
  /**
   * Style transfer buffered image.
   *
   * @param log             the log
   * @param canvasImage     the canvas image
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @return the buffered image
   */
  @Nonnull
  public BufferedImage deepDream(@Nonnull final NotebookOutput log, final BufferedImage canvasImage, final StyleSetup styleParameters, final int trainingMinutes) {
    NeuralSetup neuralSetup = new NeuralSetup(log, styleParameters).init();
    PipelineNetwork network = neuralSetup.fitnessNetwork(log);
    log.p("Input Parameters:");
    log.code(() -> {
      return toJson(styleParameters);
    });
    try {
      log.p("Input Content:");
      log.p(log.image(styleParameters.contentImage, "Content Image"));
      log.p("Style Content:");
      styleParameters.styleImages.forEach((file, styleImage) -> {
        try {
          log.p(log.image(styleImage, file));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      log.p("Input Canvas:");
      log.p(log.image(canvasImage, "Input Canvas"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    BufferedImage result = train(log, canvasImage, network, styleParameters.precision, trainingMinutes);
    try {
      log.p("Output Canvas:");
      log.p(log.image(result, "Output Canvas"));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return result;
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(@Nonnull NotebookOutput log) {
    init(log);
    Precision precision = Precision.Float;
    imageSize = 400;
    double growthFactor = Math.sqrt(1.5);
    String lakeAndForest = "H:\\SimiaCryptus\\Artistry\\Owned\\IMG_20170624_153541213-EFFECTS.jpg";
    String vanGogh = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
    String threeMusicians = "H:\\SimiaCryptus\\Artistry\\portraits\\picasso\\800px-Pablo_Picasso,_1921,_Nous_autres_musiciens_(Three_Musicians),_oil_on_canvas,_204.5_x_188.3_cm,_Philadelphia_Museum_of_Art.jpg";
    
    Map<String, StyleCoefficients> styles = new HashMap<>();
    ContentCoefficients contentCoefficients = new ContentCoefficients(
      (double) 0,
      (double) 0,
      1e-1,
      1e-1,
      1e0,
      1e1);
    double power = 0.0;
    int trainingMinutes = 90;
    
    log.h1("Phase 0");
    BufferedImage canvasImage = load(lakeAndForest, imageSize);
    //canvasImage = randomize(canvasImage);
    canvasImage = TestUtil.resize(canvasImage, imageSize, true);
    Map<String, BufferedImage> styleImages = new HashMap<>();
    styles.forEach((file, parameters) -> styleImages.put(file, load(file, file == lakeAndForest ? ((int) (imageSize * 1.5)) : imageSize)));
    BufferedImage contentImage = load(lakeAndForest, canvasImage.getWidth(), canvasImage.getHeight());
    canvasImage = deepDream(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles, power), trainingMinutes);
    for (int i = 1; i < 10; i++) {
      log.h1("Phase " + i);
      imageSize = (int) (imageSize * growthFactor);
      styleImages.clear();
      styles.forEach((file, parameters) -> styleImages.put(file, load(file, file == lakeAndForest ? ((int) (imageSize * 1.5)) : imageSize)));
      canvasImage = TestUtil.resize(canvasImage, imageSize, true);
      contentImage = load(lakeAndForest, canvasImage.getWidth(), canvasImage.getHeight());
      canvasImage = deepDream(log, canvasImage, new StyleSetup(precision, contentImage, contentCoefficients, styleImages, styles, power), trainingMinutes);
    }
    
    log.setFrontMatterProperty("status", "OK");
  }
  
  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
  }
  
  /**
   * Gets content components.
   *
   * @param node           the node
   * @param coeff_content  the coeff content
   * @param target_content the target content
   * @return the content components
   */
  public ArrayList<Tuple2<Double, DAGNode>> getContentComponents(final DAGNode node, final double coeff_content, final Tensor target_content) {
    ArrayList<Tuple2<Double, DAGNode>> functions = new ArrayList<>();
    final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
    if (coeff_content != 0) {
      functions.add(new Tuple2<>(coeff_content, network.wrap(new AvgReducerLayer(), network.wrap(new SquareActivationLayer().setAlpha(-1), node))));
    }
    return functions;
  }
  
}
