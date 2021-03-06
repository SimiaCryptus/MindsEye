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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.ActivationLayerTestBase;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer;
import com.simiacryptus.mindseye.test.SimpleEval;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Activation key apply.
 */
public abstract class ActivationLayerTest extends CudaLayerTestBase {

  /**
   * The Mode.
   */
  final ActivationLayer.Mode mode;
  private final Precision precision;
  private final int smallSize;
  private final int largeSize;

  /**
   * Instantiates a new Activation key apply.
   *
   * @param mode      the mode
   * @param precision the precision
   * @param smallSize the small size
   * @param largeSize the large size
   */
  public ActivationLayerTest(final ActivationLayer.Mode mode, final Precision precision, final int smallSize, final int largeSize) {
    this.mode = mode;
    this.precision = precision;
    this.smallSize = smallSize;
    this.largeSize = largeSize;
  }

  @Override
  public SingleDerivativeTester getDerivativeTester() {
    return new SingleDerivativeTester(1e-2, 1e-4);
  }

  @Nonnull
  @Override
  public int[][] getSmallDims(Random random) {
    return new int[][]{{smallSize, smallSize, 1}};
  }

  @Nonnull
  @Override
  public Layer getLayer(final int[][] inputSize, Random random) {
    return new ActivationLayer(mode).setPrecision(precision);
  }

  @Nonnull
  @Override
  public int[][] getLargeDims(Random random) {
    return new int[][]{
        {largeSize, largeSize, 1}
    };
  }

  @Override
  public void run(final NotebookOutput log) {
//    @Nonnull String logName = "cuda_" + log.getName() + "_all.log";
//    log.p(log.file((String) null, logName, "GPU Log"));
//    CudaSystem.addLog(new PrintStream(log.file(logName)));

    super.run(log);

    log.h3("Function Plots");
    @Nonnull final Layer layer = getLayer(new int[][]{{1, 1, 1}}, new Random());
    final List<double[]> plotData = IntStream.range(-1000, 1000).mapToDouble(x -> x / 300.0).mapToObj(x -> {
      @Nonnull Tensor input = new Tensor(new double[]{x}, 1, 1, 1);
      @Nonnull final SimpleEval eval = SimpleEval.run(layer, input);
      input.freeRef();
      @Nonnull double[] doubles = {x, eval.getOutput().get(0), eval.getDerivative()[0].get(0)};
      eval.freeRef();
      return doubles;
    }).collect(Collectors.toList());
    layer.freeRef();

    log.eval(() -> {
      return ActivationLayerTestBase.plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
    });

    log.eval(() -> {
      return ActivationLayerTestBase.plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
    });

  }

  /**
   * Configured apply double (64-bit) precision, y=x&lt;0?0:x
   */
  public static class ReLu_Double extends ActivationLayerTest {
    /**
     * Instantiates a new Re lu double.
     */
    public ReLu_Double() {
      super(ActivationLayer.Mode.RELU, Precision.Double, 2, 800);
    }

    @Override
    public Layer getReferenceLayer() {
      return new ReLuActivationLayer();
    }
  }

  /**
   * Configured apply float (32-bit) precision, y=x&lt;0?0:x
   */
  public static class ReLu_Float extends ActivationLayerTest {
    /**
     * Instantiates a new Re lu float.
     */
    public ReLu_Float() {
      super(ActivationLayer.Mode.RELU, Precision.Float, 2, 1200);
    }

    @Override
    public Layer getReferenceLayer() {
      return new ReLuActivationLayer();
    }
  }

  /**
   * Configured apply double (64-bit) precision using the Sigmoid function
   */
  public static class Sigmoid_Double extends ActivationLayerTest {
    /**
     * Instantiates a new Sigmoid double.
     */
    public Sigmoid_Double() {
      super(ActivationLayer.Mode.SIGMOID, Precision.Double, 2, 1200);
    }

    @Override
    public Layer getReferenceLayer() {
      return new SigmoidActivationLayer().setBalanced(false);
    }
  }

  /**
   * Configured apply float (32-bit) precision using the Sigmoid function
   */
  public static class Sigmoid_Float extends ActivationLayerTest {
    /**
     * Instantiates a new Sigmoid float.
     */
    public Sigmoid_Float() {
      super(ActivationLayer.Mode.SIGMOID, Precision.Float, 2, 1200);
    }

    @Override
    public Layer getReferenceLayer() {
      return new SigmoidActivationLayer().setBalanced(false);
    }
  }

}
