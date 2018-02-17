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
import com.simiacryptus.mindseye.lang.cudnn.GpuSystem;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.PrintStream;
import java.util.Random;

/**
 * The type Img concat layer run.
 */
public abstract class ImgBandSelectLayerTest extends LayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  /**
   * The LayerBase.
   */
  ImgBandSelectLayer layer;
  
  /**
   * The Input bands.
   */
  int inputBands;
  
  @Override
  public void run(NotebookOutput log) {
    @javax.annotation.Nonnull String logName = "cuda_" + log.getName() + "_all.log";
    log.p(log.file((String) null, logName, "GPU Log"));
    @javax.annotation.Nonnull PrintStream apiLog = new PrintStream(log.file(logName));
    GpuSystem.addLog(apiLog);
    super.run(log);
    apiLog.close();
    GpuSystem.apiLog.remove(apiLog);
  }
  
  /**
   * Instantiates a new Img concat layer run.
   *
   * @param precision   the precision
   * @param inputBands  the input bands
   * @param outputBands the output bands
   */
  public ImgBandSelectLayerTest(final Precision precision, int inputBands, int outputBands) {
    this.precision = precision;
    layer = new ImgBandSelectLayer(0, outputBands).setPrecision(precision);
    this.inputBands = inputBands;
  }

  @javax.annotation.Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{
      {1, 1, inputBands}
    };
  }
  
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    layer.addRef();
    return layer;
  }
  
  @javax.annotation.Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
      {64, 64, inputBands}
    };
  }
  
  @Override
  public Layer getReferenceLayer() {
    return layer.getCompatibilityLayer();
  }
  
  /**
   * Basic 64-bit run
   */
  public static class Double extends ImgBandSelectLayerTest {
    /**
     * Instantiates a new Double.
     */
    public Double() {
      super(Precision.Double, 2, 1);
    }
  }
  
  /**
   * Basic 64-bit run
   */
  public static class BigDouble extends ImgBandSelectLayerTest {
    /**
     * Instantiates a new Double.
     */
    public BigDouble() {
      super(Precision.Double, 1024, 256);
    }
  }
  
  /**
   * Basic 32-bit run
   */
  public static class Float extends ImgBandSelectLayerTest {
    /**
     * Instantiates a new Float.
     */
    public Float() {
      super(Precision.Float, 2, 1);
    }
  }
  
}
