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

package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.mindseye.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.util.data.DoubleStatistics;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import javax.annotation.Nonnull;
import java.util.Comparator;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class PCAUtil {
  /**
   * Forked from Apache Commons Math
   *
   * @param stream the stream
   * @return covariance covariance
   */
  @Nonnull
  public static RealMatrix getCovariance(@Nonnull final Supplier<Stream<double[]>> stream) {
    final int dimension = stream.get().findAny().get().length;
    final List<DoubleStatistics> statList = IntStream.range(0, dimension * dimension)
      .mapToObj(i -> new DoubleStatistics()).collect(Collectors.toList());
    stream.get().forEach(array -> {
      for (int i = 0; i < dimension; i++) {
        for (int j = 0; j <= i; j++) {
          statList.get(i * dimension + j).accept(array[i] * array[j]);
        }
      }
      RecycleBin.DOUBLES.recycle(array, array.length);
    });
    @Nonnull final RealMatrix covariance = new BlockRealMatrix(dimension, dimension);
    for (int i = 0; i < dimension; i++) {
      for (int j = 0; j <= i; j++) {
        final double v = statList.get(i * dimension + j).getAverage();
        covariance.setEntry(i, j, v);
        covariance.setEntry(j, i, v);
      }
    }
    return covariance;
  }
  
  public static Tensor[] pcaFeatures_0(final RealMatrix covariance, final int components, final int[] featureDimensions) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(covariance);
    final int[] orderedVectors = IntStream.range(0, components).mapToObj(x -> x)
      .sorted(Comparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
    return IntStream.range(0, orderedVectors.length)
      .mapToObj(i -> {
          @Nonnull final Tensor src = new Tensor(decomposition.getEigenvector(orderedVectors[i]).toArray(), featureDimensions).copy();
          return src.scale(1.0 / src.rms());
        }
      ).toArray(i -> new Tensor[i]);
  }
  
  public static Tensor[] pcaFeatures_inv(final RealMatrix covariance, final int components, final int[] featureDimensions) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(covariance);
    final int[] orderedVectors = IntStream.range(0, components).mapToObj(x -> x)
      .sorted(Comparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
    return IntStream.range(0, orderedVectors.length)
      .mapToObj(i -> {
          @Nonnull final Tensor src = new Tensor(decomposition.getEigenvector(orderedVectors[i]).toArray(), featureDimensions).copy();
          return src
            .scale(1.0 / src.rms())
            .scale((decomposition.getRealEigenvalue(orderedVectors[i]) / decomposition.getRealEigenvalue(orderedVectors[0])))
            ;
        }
      ).toArray(i -> new Tensor[i]);
  }
  
  public static Tensor[] pcaFeatures_norm(final RealMatrix covariance, final int components, final int[] featureDimensions) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(covariance);
    final int[] orderedVectors = IntStream.range(0, components).mapToObj(x -> x)
      .sorted(Comparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
    return IntStream.range(0, orderedVectors.length)
      .mapToObj(i -> {
          @Nonnull final Tensor src = new Tensor(decomposition.getEigenvector(orderedVectors[i]).toArray(), featureDimensions).copy();
          return src
            .scale(1.0 / src.rms())
            .scale((decomposition.getRealEigenvalue(orderedVectors[orderedVectors.length - 1]) / decomposition.getRealEigenvalue(orderedVectors[i])))
            ;
        }
      ).toArray(i -> new Tensor[i]);
  }
  
  public static void populatePCAKernel(final Tensor kernel, final Tensor[] featureSpaceVectors) {
    final int outputBands = featureSpaceVectors.length;
    @Nonnull final int[] filterDimensions = kernel.getDimensions();
    kernel.setByCoord(c -> {
      final int kband = c.getCoords()[2];
      final int outband = kband % outputBands;
      final int inband = (kband - outband) / outputBands;
      assert outband < outputBands;
      int x = c.getCoords()[0];
      int y = c.getCoords()[1];
      x = filterDimensions[0] - (x + 1);
      y = filterDimensions[1] - (y + 1);
      final double v = featureSpaceVectors[inband].get(x, y, outband);
      return Double.isFinite(v) ? v : kernel.get(c);
    });
  }
}
