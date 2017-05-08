package com.simiacryptus.mindseye.test.regression;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.mindseye.net.activation.SigmoidActivationLayer;
import com.simiacryptus.mindseye.net.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.net.activation.SqActivationLayer;
import com.simiacryptus.mindseye.net.basic.BiasLayer;
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerGPU;
import com.simiacryptus.mindseye.net.dev.DenseSynapseLayerJBLAS;
import com.simiacryptus.mindseye.net.dev.ToeplitzSynapseLayerJBLAS;
import com.simiacryptus.mindseye.net.loss.MeanSqLossLayer;
import com.simiacryptus.mindseye.net.media.MaxSubsampleLayer;
import com.simiacryptus.mindseye.net.reducers.ProductLayer;
import com.simiacryptus.mindseye.net.reducers.SumInputsLayer;
import com.simiacryptus.mindseye.net.reducers.SumReducerLayer;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

public class ComponentPerformanceTests {
  public static final double deltaFactor = 1e-6;

  private static final Logger log = LoggerFactory.getLogger(ComponentPerformanceTests.class);

  public static int test(final NNLayer<?> component, final Tensor outputPrototype, final Tensor... inputPrototype) throws Throwable {
    long timeout = System.currentTimeMillis() + TimeUnit.SECONDS.toMillis(60);
    int iterations = 0;
    while(timeout > System.currentTimeMillis()) {
      iterations++;
      NNResult eval = component.eval(inputPrototype);
      DeltaSet deltaSet = new DeltaSet();
      eval.accumulate(deltaSet, new Tensor[]{outputPrototype});
    }
    log.info(String.format("Iterations completed: %s", iterations));
    return iterations;
  }

  @org.junit.Test
  public void testBiasLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new BiasLayer(outputPrototype.getDims()).setWeights(i -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayer1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testToeplitzSynapseLayerJBLAS1() throws Throwable {
    final Tensor inputPrototype = new Tensor(3,3).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = new Tensor(3,3);
    final NNLayer<?> component = new ToeplitzSynapseLayerJBLAS(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testToeplitzSynapseLayerJBLAS2() throws Throwable {
    final Tensor inputPrototype = new Tensor(3,3).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = new Tensor(3,3,2,3);
    final NNLayer<?> component = new ToeplitzSynapseLayerJBLAS(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayerJBLAS1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayerJBLAS(inputPrototype.getDims(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayerGPU1() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayerGPU(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testDenseSynapseLayer2() throws Throwable {
    final Tensor outputPrototype = new Tensor(2);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new DenseSynapseLayer(inputPrototype.dim(), outputPrototype.getDims()).setWeights(() -> Util.R.get().nextGaussian());
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testMaxSubsampleLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1, 1, 1);
    final Tensor inputPrototype = new Tensor(2, 2, 1).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new MaxSubsampleLayer(2, 2, 1);
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testProductLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new ProductLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSigmoidLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SigmoidActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSoftmaxLayer() throws Throwable {
    final Tensor inputPrototype = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor outputPrototype = inputPrototype.copy();
    final NNLayer<?> component = new SoftmaxActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSqActivationLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(3);
    final Tensor inputPrototype = new Tensor(3).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SqActivationLayer();
    test(component, outputPrototype, inputPrototype);
  }

  @org.junit.Test
  public void testSqLossLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new MeanSqLossLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSumLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SumInputsLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

  @org.junit.Test
  public void testSumReducerLayer() throws Throwable {
    final Tensor outputPrototype = new Tensor(1);
    final Tensor inputPrototype1 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final Tensor inputPrototype2 = new Tensor(2).fill(() -> Util.R.get().nextGaussian());
    final NNLayer<?> component = new SumReducerLayer();
    test(component, outputPrototype, inputPrototype1, inputPrototype2);
  }

}
