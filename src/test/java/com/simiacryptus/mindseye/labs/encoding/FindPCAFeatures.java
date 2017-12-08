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

package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.io.NotebookOutput;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

/**
 * The type Find feature space.
 */
class FindPCAFeatures extends FindFeatureSpace {
  
  /**
   * Instantiates a new Find feature space.
   *
   * @param log        the log
   * @param inputBands the input bands
   * @param features   the features
   */
  public FindPCAFeatures(NotebookOutput log, int inputBands, Tensor[][] features) {
    super(log, inputBands, features);
  }
  
  /**
   * Invoke find feature space.
   *
   * @return the find feature space
   */
  @Override
  public FindFeatureSpace invoke() {
    averages = findBandBias(features);
    Tensor[][] featureVectors = Arrays.stream(features).map(tensor -> {
      return new Tensor[]{tensor[0], tensor[1].mapCoords((v, c) -> v - averages[c.coords[2]])};
    }).toArray(i -> new Tensor[i][]);
    vectors = findFeatureSpace(log, featureVectors, inputBands);
    return this;
  }
  
  /**
   * Find band bias double [ ].
   *
   * @param features the features
   * @return the double [ ]
   */
  protected double[] findBandBias(Tensor[][] features) {
    Tensor prototype = features[0][1];
    int[] dimensions = prototype.getDimensions();
    int outputBands = dimensions[2];
    return IntStream.range(0, outputBands).parallel().mapToDouble(b -> {
      return Arrays.stream(features).mapToDouble(tensor -> {
        return Arrays.stream(tensor[1].mapCoords((v, c) -> c.coords[2] == b ? v : Double.NaN).getData()).filter(Double::isFinite).average().getAsDouble();
      }).average().getAsDouble();
    }).toArray();
  }
  
  /**
   * Find feature space tensor [ ].
   *
   * @param log            the log
   * @param featureVectors the feature vectors
   * @param components     the components
   * @return the tensor [ ]
   */
  protected Tensor[] findFeatureSpace(NotebookOutput log, Tensor[][] featureVectors, int components) {
    return log.code(() -> {
      int column = 1;
      int[] dimensions = featureVectors[0][column].getDimensions();
      double[][] data = Arrays.stream(featureVectors).map(x -> x[column].getData()).toArray(i -> new double[i][]);
      RealMatrix realMatrix = MatrixUtils.createRealMatrix(data);
      Covariance covariance = new Covariance(realMatrix);
      RealMatrix covarianceMatrix = covariance.getCovarianceMatrix();
      EigenDecomposition decomposition = new EigenDecomposition(covarianceMatrix);
      int[] orderedVectors = IntStream.range(0, components).mapToObj(x -> x)
        .sorted(Comparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
      return IntStream.range(0, orderedVectors.length)
        .mapToObj(i -> {
            Tensor src = new Tensor(decomposition.getEigenvector(orderedVectors[i]).toArray(), dimensions).copy();
            return src
              .scaleInPlace(1.0 / src.rms())
              .scale((decomposition.getRealEigenvalue(orderedVectors[i]) / decomposition.getRealEigenvalue(orderedVectors[inputBands-1])))
              //.scale(decomposition.getRealEigenvalue(orderedVectors[inputBands-1]) / decomposition.getRealEigenvalue(orderedVectors[i]))
              //.scale((1.0 / decomposition.getRealEigenvalue(orderedVectors[0])))
              .scaleInPlace(Math.sqrt(6. / (components + featureVectors[0][column].dim() + 1)))
              ;
          }
        ).toArray(i -> new Tensor[i]);
    });
  }
}
