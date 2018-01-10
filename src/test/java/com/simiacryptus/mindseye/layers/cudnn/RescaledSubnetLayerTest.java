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
import org.junit.Ignore;
import org.junit.Test;

import java.io.PrintStream;
import java.util.Random;

/**
 * The type Rescaled subnet layer run.
 */
public abstract class RescaledSubnetLayerTest extends LayerTestBase // CudnnLayerTestBase
{
  
  /**
   * The Convolution layer.
   */
  ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 1, 1);
  
  public RescaledSubnetLayerTest() {
  
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
  public int[][] getInputDims(Random random) {
    return new int[][]{
      {8, 8, 1}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return new RescaledSubnetLayer(2, convolutionLayer.set(this::random));
  }
  
  @Override
  public Class<? extends NNLayer> getReferenceLayerClass() {
    return com.simiacryptus.mindseye.layers.java.RescaledSubnetLayer.class;
  }
  
  /**
   * Basic Test
   */
  public static class Basic extends RescaledSubnetLayerTest {
    @Override
    @Test(timeout = 15 * 60 * 1000)
    @Ignore // Crashing Bug!?!?
    public void test() throws Throwable {
      super.test();
    }
  }
  
}
