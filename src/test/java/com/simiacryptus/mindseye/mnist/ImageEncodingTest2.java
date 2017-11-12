/*
 * Copyright (c) 2017 by Andrew Charneski.
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

package com.simiacryptus.mindseye.mnist;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f64.ImgBandBiasLayer;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ImageEncodingTest2 extends ImageEncodingTest {

  @Override
  @Test
  @Category(TestCategories.Report.class)
  public void test() throws Exception {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != out) ((MarkdownNotebookOutput) log).addCopy(out);
      
      int pretrainMinutes = 1;
      int timeoutMinutes = 1;
      int size = 256;
      int images = 10;
      
      Tensor[][] trainingImages = getImages(log, size, 100, "kangaroo");
      
      log.h1("First Layer");
      InitializationStep step0 = log.code(()->{
        return new InitializationStep(log, trainingImages,
          images, size, pretrainMinutes, timeoutMinutes, 3, 7, 5);
      }).invoke();
      
    }
  }
  
  @Override
  protected void initialize(NotebookOutput log, Tensor[] features, ConvolutionLayer convolutionLayer, ImgBandBiasLayer biasLayer) {
    Tensor prototype = features[0];
    int[] dimensions = prototype.getDimensions();
    int[] filterDimensions = convolutionLayer.filter.getDimensions();
    assert filterDimensions[0] == dimensions[0];
    assert filterDimensions[1] == dimensions[1];
    int outputBands = dimensions[2];
    assert outputBands == biasLayer.getBias().length;
    int inputBands = filterDimensions[2] / outputBands;
    double[] averages = IntStream.range(0, outputBands).parallel().mapToDouble(b -> {
      return Arrays.stream(features).mapToDouble(tensor -> {
        return Arrays.stream(tensor.mapCoords((v, c) -> c.coords[2] == b ? v : Double.NaN).getData()).filter(Double::isFinite).average().getAsDouble();
      }).average().getAsDouble();
    }).toArray();
    biasLayer.setWeights(i -> {
      double v = averages[i];
      return Double.isFinite(v) ? v : biasLayer.getBias()[i];
    });
    double[][] data = Arrays.stream(features).map(tensor -> {
      return tensor.mapCoords((v, c) -> v - averages[c.coords[2]]);
    }).map(x -> x.getData()).toArray(i -> new double[i][]);
    log.code(() -> {
      RealMatrix realMatrix = MatrixUtils.createRealMatrix(data);
      Covariance covariance = new Covariance(realMatrix);
      RealMatrix covarianceMatrix = covariance.getCovarianceMatrix();
      EigenDecomposition decomposition = new EigenDecomposition(covarianceMatrix);
      int[] orderedVectors = IntStream.range(0, inputBands).mapToObj(x -> x)
        .sorted(Comparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
      List<Tensor> rawComponents = IntStream.range(0, orderedVectors.length)
        .mapToObj(i -> {
            Tensor src = new Tensor(decomposition.getEigenvector(orderedVectors[i]).toArray(), dimensions[0], dimensions[1], outputBands).copy();
            return src
              .scale(1.0 / src.rms())
              //.scale((decomposition.getRealEigenvalue(orderedVectors[inputBands-1])))
              //.scale((decomposition.getRealEigenvalue(orderedVectors[i])))
              //.scale((1.0 / decomposition.getRealEigenvalue(orderedVectors[0])))
              .scale(Math.sqrt(6. / (inputBands + convolutionLayer.filter.dim() + 1)))
              ;
          }
        )
        .collect(Collectors.toList());
      Tensor[] vectors = rawComponents.stream().toArray(i -> new Tensor[i]);
      convolutionLayer.filter.fillByCoord(c -> {
        int outband = c.coords[2] / inputBands;
        int inband = c.coords[2] % inputBands;
        assert c.coords[0] < dimensions[0];
        assert c.coords[1] < dimensions[1];
        assert outband < outputBands;
        assert inband < inputBands;
        double v = vectors[inband].get(dimensions[0] - (c.coords[0] + 1), dimensions[1] - (c.coords[1] + 1), outputBands - (outband + 1));
        return Double.isFinite(v) ? v : convolutionLayer.filter.get(c);
      });
    });
  }
  
}
