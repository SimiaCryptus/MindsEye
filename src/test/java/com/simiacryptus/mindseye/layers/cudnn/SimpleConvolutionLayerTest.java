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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CuDNN;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.*;
import com.simiacryptus.util.io.NotebookOutput;

import java.io.PrintStream;
import java.util.Random;

/**
 * The type Simple convolution layer apply.
 */
public abstract class SimpleConvolutionLayerTest extends CudnnLayerTestBase {
  
  /**
   * The Radius.
   */
  public final int radius;
  /**
   * The Bands.
   */
  public final int bands;
  /**
   * The Layer.
   */
  SimpleConvolutionLayer layer;
  
  
  /**
   * Instantiates a new Simple convolution layer apply.
   *
   * @param radius    the radius
   * @param bands     the bands
   * @param precision the precision
   */
  protected SimpleConvolutionLayerTest(final int radius, final int bands, final Precision precision) {
    this.radius = radius;
    this.bands = bands;
    layer = new SimpleConvolutionLayer(radius, radius, bands * bands).setPrecision(precision);
    layer.kernel.set(() -> random());
  }
  
  @Override
  public int[][] getInputDims(Random random) {
    return new int[][]{
      {radius, radius, bands}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize, Random random) {
    return layer;
  }
  
  @Override
  public int[][] getPerfDims(Random random) {
    return new int[][]{
      {100, 100, bands}
    };
  }
  
  @Override
  public NNLayer getReferenceLayer() {
    final ConvolutionLayer convolutionLayer = new ConvolutionLayer(radius, radius, bands, bands, true);
    final Tensor tensor = new Tensor(layer.kernel.getDimensions());
    tensor.setByCoord(c -> {
      final int band = c.getCoords()[2];
      final int bandX = band % bands;
      final int bandY = (band - bandX) / bands;
      assert band == bandX + bandY * bands;
      final int bandT = bandY + bandX * bands;
      return layer.kernel.get(c.getCoords()[0], c.getCoords()[1], bandT);
    });
    convolutionLayer.kernel.set(tensor);
    return convolutionLayer;
  }
  
  /**
   * Maximally-basic single-value "convolution" in 64 bits
   */
  public static class Basic extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Image.
     */
    public Basic() {
      super(1, 1, Precision.Double);
    }
  }
  
  /**
   * Typical 3x3 image convolution (64-bit)
   */
  public static class Image extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Image.
     */
    public Image() {
      super(3, 3, Precision.Double);
    }
  }
  
  /**
   * Typical 3x3 image convolution (32-bit)
   */
  public static class Image_Float extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Image float.
     */
    public Image_Float() {
      super(3, 3, Precision.Float);
    }
  
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  
  }
  
  /**
   * Basic single-band 3x3 image filter.
   */
  public static class Matrix extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Matrix.
     */
    public Matrix() {
      super(3, 1, Precision.Double);
    }
  }
  
  /**
   * The type Temp.
   */
  public static class Temp extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Matrix.
     */
    public Temp() {
      super(1, 3, Precision.Double);
    }
  }
  
  /**
   * Basic multi-band, 1-pixel-radius filter.
   */
  public static class MultiBand extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Multi band.
     */
    public MultiBand() {
      super(1, 3, Precision.Double);
    }
  }
  
  /**
   * Base allocationOverflow configuration demonstrating the absence of failure in this case.
   */
  public static class Bug_Control extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Multi band.
     */
    public Bug_Control() {
      super(3, 8, Precision.Double);
      validateDifferentials = false;
    }
    
    @Override
    public void run(NotebookOutput log) {
      String logName = "cuda_" + log.getName() + "_all.log";
      log.p(log.file((String) null, logName, "GPU Log"));
      PrintStream apiLog = new PrintStream(log.file(logName));
      CuDNN.addLog(apiLog);
      super.run(log);
      apiLog.close();
      CuDNN.apiLog.remove(apiLog);
    }
    
    
    @Override
    public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
      return null;
    }
    
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      return new PerformanceTester().setBatches(10).setSamples(1);
    }
    
    protected ComponentTest<ToleranceStatistics> getReferenceIOTester() {
      return new ReferenceIO(getReferenceIO());
    }
    
  }
  
  /**
   * Demonstration of a suspected CuDNN bug when using 0 padding with the GPU convolution operation.
   */
  public static class Bug extends Bug_Control {
    /**
     * Instantiates a new Multi band.
     */
    public Bug() {
      super();
      layer.setPaddingXY(0, 0);
    }
    
  }
  
  /**
   * Simple 256x256 band 1-pixel "convolution"
   */
  public static class Big extends SimpleConvolutionLayerTest {
    /**
     * Instantiates a new Multi band.
     */
    public Big() {
      super(3, 128, Precision.Double);
      validateDifferentials = false;
    }
    
    @Override
    public ComponentTest<TrainingTester.ComponentResult> getTrainingTester() {
//      return null;
      return super.getTrainingTester();
    }
    
    @Override
    public NNLayer getReferenceLayer() {
      return null;
    }
    
    @Override
    public int[][] getPerfDims(Random random) {
      return new int[][]{
        {30, 30, bands}
      };
    }
    
    public ComponentTest<ToleranceStatistics> getPerformanceTester() {
      ComponentTest<ToleranceStatistics> inner = new PerformanceTester().setBatches(10);
      return (log1, component, inputPrototype) -> {
        String logName = "cuda_" + log1.getName() + "_perf.log";
        PrintStream apiLog = null;
        try {
          apiLog = new PrintStream(log1.file(logName));
          CuDNN.addLog(apiLog);
          return inner.test(log1, component, inputPrototype);
        } finally {
          log1.p(log1.file((String) null, logName, "GPU Log"));
          if (null != apiLog) {
    
            apiLog.close();
            CuDNN.apiLog.remove(apiLog);
          }
        }
      };
    }
    
  }
  
}
