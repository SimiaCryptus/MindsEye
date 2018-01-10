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
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import org.junit.Assert;
import org.junit.Test;

/**
 * The type Convolution network run.
 */
public abstract class ConvolutionNetworkTest extends CudnnLayerTestBase {
  
  final int radius;
  final int inputBands;
  final int outputBands;
  /**
   * The Precision.
   */
  final Precision precision;
  final ConvolutionLayer convolutionLayer;
  NNLayer layer;
  
  /**
   * Instantiates a new Convolution layer run.
   *
   * @param radius      the radius
   * @param inputBands  the input bands
   * @param outputBands the output bands
   * @param precision   the precision
   */
  protected ConvolutionNetworkTest(final int radius, final int inputBands, final int outputBands, final Precision precision) {
    this.radius = radius;
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    this.convolutionLayer = new ConvolutionLayer(radius, radius, inputBands, outputBands).setPrecision(precision);
    convolutionLayer.getKernel().set(() -> random());
    this.precision = precision;
    this.layer = convolutionLayer.explode();
  }
  
  //
//  @Override
//  public void run(NotebookOutput log) {
//    String logName = "cuda_" + log.getName() + "_all.log";
//    log.p(log.file((String) null, logName, "GPU Log"));
//    PrintStream added = new PrintStream(log.file(logName));
//    CuDNN.apiLog.add(added);
//    super.run(log);
//    CuDNN.apiLog.remove(added);
//  }
//
  @Test
  public void verifyWeights() {
    ExplodedConvolutionGrid explodedNetwork = this.convolutionLayer.getExplodedNetwork();
    int[] kernelDims = this.convolutionLayer.getKernel().getDimensions();
    Tensor testData = new Tensor(kernelDims).map(x -> Math.random());
    explodedNetwork.write(testData);
    Tensor echo = explodedNetwork.extractKernel();
    Assert.assertEquals(testData, echo);
  }
  
  
  @Override
  public int[][] getInputDims() {
    return new int[][]{
      {5, 5, inputBands}
    };
  }
  
  @Override
  public NNLayer getLayer(final int[][] inputSize) {
    return this.layer;
  }
  
  @Override
  public int[][] getPerfDims() {
    return new int[][]{
      {100, 100, inputBands}
    };
  }
  
  /**
   * Expands the an example low-level network implementing general convolutions. (64-bit)
   */
  public static class DoubleConvolutionNetwork extends ConvolutionNetworkTest {
    /**
     * Instantiates a new Double.
     */
    public DoubleConvolutionNetwork() {
      super(1, 2, 1, Precision.Double);
      convolutionLayer.setBatchBands(1);
      this.layer = convolutionLayer.explode();
    }
  
  
    @Override
    public int[][] getInputDims() {
      return new int[][]{
        {1, 1, inputBands}
      };
    }
  
    @Override
    public int[][] getPerfDims() {
      return getInputDims();
    }
  
  }
  
  /**
   * Expands the an example low-level network implementing general convolutions. (64-bit)
   */
  public static class Tree extends ConvolutionNetworkTest {
    /**
     * Instantiates a new Double.
     */
    public Tree() {
      super(1, 4, 4, Precision.Double);
      convolutionLayer.setBatchBands(2);
    }
  }
  
  /**
   * Expands the an example low-level network implementing general convolutions. (64-bit)
   */
  public static class BigDoubleConvolutionNetwork extends ConvolutionNetworkTest {
    /**
     * Instantiates a new Double.
     */
    public BigDoubleConvolutionNetwork() {
      super(3, 4, 20, Precision.Double);
      convolutionLayer.setBatchBands(2);
      layer = convolutionLayer.explode();
      validateDifferentials = false;
  
    }
  
    @Override
    public NNLayer getReferenceLayer() {
      return null;
    }
  }
  
  /**
   * Expands the an example low-level network implementing general convolutions. (32-bit)
   */
  public static class FloatConvolutionNetwork extends ConvolutionNetworkTest {
    /**
     * Instantiates a new Float.
     */
    public FloatConvolutionNetwork() {
      super(3, 4, 8, Precision.Float);
    }
  
    @Override
    public SingleDerivativeTester getDerivativeTester() {
      return new SingleDerivativeTester(1e-2, 1e-3);
    }
  
  }
  
  
}
