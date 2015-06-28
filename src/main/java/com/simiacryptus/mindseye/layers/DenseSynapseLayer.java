package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.DeltaInversionBuffer;
import com.simiacryptus.mindseye.learning.DeltaMassMomentumBuffer;
import com.simiacryptus.mindseye.learning.NNResult;

public class DenseSynapseLayer extends NNLayer implements MassParameters<DenseSynapseLayer> {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);

  private double backpropPruning = 0.;
  private DeltaInversionBuffer deltaBuffer;
  private boolean frozen = false;
  private DeltaMassMomentumBuffer massMomentum;
  private final int[] outputDims;
  
  public final NDArray weights;

  public DenseSynapseLayer(final int inputs, final int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputDims));
    this.massMomentum = new DeltaMassMomentumBuffer(this.weights);
    this.deltaBuffer = new DeltaInversionBuffer(1, this.massMomentum);
  }

  public DenseSynapseLayer addWeights(final DoubleSupplier f) {
    for (int i = 0; i < this.weights.data.length; i++)
    {
      this.weights.data[i] += f.getAsDouble();
    }
    return this;
  }

  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray output = new NDArray(this.outputDims);
    final NDArray inputGradient = new NDArray(input.dim(), output.dim());
    final NDArray weightGradient = this.frozen ? null : new NDArray(this.weights.dim(), output.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(o -> {
        final double a = this.weights.get(i, o);
        final double b = input.data[i];
        inputGradient.add(new int[] { i, o }, a);
        if (null != weightGradient) {
          weightGradient.add(new int[] { this.weights.index(i, o), o }, b);
        }
        output.add(o, b * a);
      });
    });
    return new NNResult(output) {
      
      @Override
      public void feedback(final NDArray data) {
        synchronized (DenseSynapseLayer.this) {
          if (null != weightGradient) {
            DenseSynapseLayer.this.deltaBuffer.feed(weightGradient, data.data);
          }
          if (inObj.isAlive()) {
            final double[] delta = data.data;
            final int[] dims = inputGradient.getDims();
            final double[] inverted = org.jblas.Solve.solveLeastSquares(
                new DoubleMatrix(dims[0], inputGradient.getDims()[1], inputGradient.data).transpose(),
                new DoubleMatrix(delta.length, 1, delta)).data;
            final double[] mcdelta = Arrays.copyOf(inverted, inverted.length);
            for (int i = 0; i < mcdelta.length; i++) {
              mcdelta[i] *= Math.random() < DenseSynapseLayer.this.backpropPruning ? 0 : 1;
            }
            inObj.feedback(new NDArray(inObj.data.getDims(), mcdelta));
          }
        }
      }

      @Override
      public boolean isAlive() {
        return null != weightGradient || inObj.isAlive();
      }
    };
  }

  public DenseSynapseLayer freeze() {
    return freeze(true);
  }

  public DenseSynapseLayer freeze(final boolean b) {
    this.frozen = b;
    return this;
  }

  public double getBackpropPruning() {
    return this.backpropPruning;
  }

  @Override
  public double getMass() {
    return this.massMomentum.getMass();
  }

  @Override
  public double getMomentumDecay() {
    return this.massMomentum.getMomentumDecay();
  }

  public DenseSynapseLayer setBackpropPruning(final double backpropPruning) {
    this.backpropPruning = backpropPruning;
    return this;
  }
  
  @Override
  public DenseSynapseLayer setMass(final double mass) {
    this.massMomentum.setMass(mass);
    return this;
  }
  
  @Override
  public DenseSynapseLayer setMomentumDecay(final double momentumDecay) {
    this.massMomentum.setMomentumDecay(momentumDecay);
    return this;
  }
  
  public DenseSynapseLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.data, i -> f.getAsDouble());
    return this;
  }
  
  public DenseSynapseLayer thaw() {
    return freeze(false);
  }

}
