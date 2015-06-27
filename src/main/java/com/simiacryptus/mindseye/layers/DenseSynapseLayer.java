package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.NNResult;

public class DenseSynapseLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);
  
  private final int[] outputDims;
  public final NDArray weights;
  private boolean frozen = false;
  private DeltaInversionBuffer deltaBuffer;
  double backpropPruning = 0.2;
  private double momentumDecay = 0.9;
  
  public DenseSynapseLayer(int inputs, int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputDims));
  }
  
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray output = new NDArray(outputDims);
    final NDArray inputGradient = new NDArray(input.dim(), output.dim());
    final NDArray weightGradient = this.frozen ? null : new NDArray(weights.dim(), output.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(o -> {
        double a = weights.get(i, o);
        double b = input.data[i];
        inputGradient.add(new int[] { i, o }, a);
        if (null != weightGradient) weightGradient.add(new int[] { weights.index(i, o), o }, b);
        output.add(o, b * a);
      });
    });
    return new NNResult(output) {

      @Override
      public void feedback(NDArray data) {
        synchronized (DenseSynapseLayer.this) {
          if(null == deltaBuffer)
          {
            deltaBuffer = new DeltaInversionBuffer(1, new DeltaMomentumBuffer(weights).setDecay(momentumDecay));
          }
          if (null != weightGradient) {
            deltaBuffer.feed(weightGradient, data.data);
          }
          if (inObj.isAlive()) {
            double[] delta = data.data;
            int[] dims = inputGradient.getDims();
            double[] inverted = org.jblas.Solve.solveLeastSquares(
                new DoubleMatrix(dims[0], inputGradient.getDims()[1], inputGradient.data).transpose(),
                new DoubleMatrix(delta.length, 1, delta)).data;
            double[] mcdelta = Arrays.copyOf(inverted, inverted.length);
            for (int i = 0; i < mcdelta.length; i++) {
              mcdelta[i] *= (Math.random() < backpropPruning) ? 0 : 1;
            }
            inObj.feedback(new NDArray(inObj.data.getDims(), mcdelta));
          }
        }
      }
      
      public boolean isAlive() {
        return null != weightGradient || inObj.isAlive();
      }
    };
  }
  
  public DenseSynapseLayer addWeights(DoubleSupplier f) {
    for (int i = 0; i < weights.data.length; i++)
    {
      weights.data[i] += f.getAsDouble();
    }
    return this;
  }
  
  public DenseSynapseLayer setWeights(DoubleSupplier f) {
    Arrays.parallelSetAll(weights.data, i -> f.getAsDouble());
    return this;
  }
  
  public DenseSynapseLayer freeze() {
    return freeze(true);
  }
  
  public DenseSynapseLayer thaw() {
    return freeze(false);
  }
  
  public DenseSynapseLayer freeze(boolean b) {
    this.frozen = b;
    return this;
  }

  public double getMomentumDecay() {
    return momentumDecay;
  }

  public DenseSynapseLayer setMomentumDecay(double momentumDecay) {
    this.momentumDecay = momentumDecay;
    return this;
  }
  
}
