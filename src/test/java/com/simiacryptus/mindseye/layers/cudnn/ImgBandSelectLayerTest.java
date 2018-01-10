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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.LayerTestBase;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.PrintStream;

/**
 * The type Img concat layer run.
 */
public abstract class ImgBandSelectLayerTest extends LayerTestBase {
  
  /**
   * The Precision.
   */
  final Precision precision;
  /**
   * The Layer.
   */
  ImgBandSelectLayer layer = new ImgBandSelectLayer(1, 2);
  
  /**
   * Instantiates a new Img concat layer run.
   *
   * @param precision the precision
   */
  public ImgBandSelectLayerTest(final Precision precision) {
    this.precision = precision;
  }
  
  @Override
  public void run(NotebookOutput log) {
    String logName = "cuda_" + log.getName() + "_all.log";
    log.p(log.file((String) null, logName, "GPU Log"));
    PrintStream apiLog = new PrintStream(log.file(logName));
    CuDNN.apiLog.add(apiLog);
    super.run(log);
    apiLog.close();
    CuDNN.apiLog.remove(apiLog);
  }
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {1, 1, 2}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return layer;
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {64, 64, 2}
    };
  }
  
  @Override
  public NNLayer getReferenceLayer() {
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
      super(Precision.Double);
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
      super(Precision.Float);
    }
  }
  
}
