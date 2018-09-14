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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.util.io.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.Random;

/**
 * The type Img eval layer apply.
 */
public abstract class ImgBandSelectLayerTest extends CudaLayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  private final int smallSize;
  private final int largeSize;
  /**
   * The LayerBase.
   */
  ImgBandSelectLayer layer;
  /**
   * The Input bands.
   */
  int inputBands;
  
  /**
   * Instantiates a new Img eval layer apply.
   *
   * @param precision  the precision
   * @param inputBands the input bands
   * @param fromBand   the from band
   * @param toBand     the output bands
   */
  public ImgBandSelectLayerTest(final Precision precision, int inputBands, final int fromBand, int toBand) {
    this.precision = precision;
    layer = new ImgBandSelectLayer(fromBand, toBand).setPrecision(precision);
    this.inputBands = inputBands;
    smallSize = 2;
    largeSize = 1000;
    testingBatchSize = 1;
  }
  
  @Override
  public void run(NotebookOutput log) {
//    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//    log.p(log.file((String) null, logName, "GPU Log"));
//    @Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
//    CudaSystem.addLog(apiLog);
    super.run(log);
//    apiLog.close();
//    CudaSystem.apiLog.remove(apiLog);
  }
  
  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {smallSize, smallSize, inputBands}
    };
  }
  
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    layer.addRef();
    return layer;
  }
  
  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
      {largeSize, largeSize, inputBands}
    };
  }
  
  @Override
  public Layer getReferenceLayer() {
    return layer.getCompatibilityLayer();
  }
  
  /**
   * Basic 64-bit apply
   */
  public static class Double extends ImgBandSelectLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double, 5, 2, 4);
    }
  }

//  /**
//   * Basic 64-bit apply
//   */
//  public static class BigDouble extends ImgBandSelectLayerTest {
//    /**
//     * Instantiates a new Double.
//     */
//    public BigDouble() {
//      super(Precision.Double, 1024, 0, 256);
//    }
//  }
  
  /**
   * Basic 32-bit apply
   */
  public static class Float extends ImgBandSelectLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float, 2, 0, 1);
    }
  }
  
}
