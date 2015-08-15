package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaFlushBuffer;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.GradientDescentAccumulator;
import com.simiacryptus.mindseye.learning.NNResult;

public class DenseSynapseLayer extends NNLayer implements DeltaTransaction {
  private final class DenseSynapseResult extends NNResult {
    private final NNResult inObj;
    private final NDArray inputGradient;
    private final NDArray weightGradient;
    
    private DenseSynapseResult(final NDArray data, final NDArray inputGradient, final NDArray weightGradient, final NNResult inObj) {
      super(data);
      if (null == inputGradient) throw new IllegalArgumentException();
      // if(null == weightGradient) throw new IllegalArgumentException();
      this.inputGradient = inputGradient;
      this.weightGradient = weightGradient;
      this.inObj = inObj;
    }
    
    @Override
    public void feedback(final NDArray data) {
      NDArray passback = null;
      if (null != this.weightGradient) {
        DenseSynapseLayer.this.deltaBuffer.feed(this.weightGradient, data.getData());
      }
      if (this.inObj.isAlive()) {
        final double[] delta = data.getData();
        DoubleMatrix pseudoinverse;
        try {
          pseudoinverse = NDArray.inverseCache.get(this.inputGradient);
        } catch (final ExecutionException e) {
          throw new RuntimeException(e);
        }
        final double[] inverted = pseudoinverse.mmul(new DoubleMatrix(delta.length, 1, delta)).data;
        final double[] mcdelta = Arrays.copyOf(inverted, inverted.length);
        for (int i = 0; i < mcdelta.length; i++) {
          mcdelta[i] *= Math.random() < DenseSynapseLayer.this.backpropPruning ? 0 : 1;
        }
        passback = new NDArray(this.inObj.data.getDims(), mcdelta);
      }
      if (null != passback)
      {
        this.inObj.feedback(passback);
      }
      if (isVerbose()) {
        DenseSynapseLayer.log.debug(String.format("Feed back: %s => %s", data, passback));
      }
    }
    
    @Override
    public boolean isAlive() {
      return null != this.weightGradient || this.inObj.isAlive();
    }
  }
  
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);
  
  public static int[] transpose(final int[] dims2) {
    final int[] dims = new int[] { dims2[1], dims2[0] };
    return dims;
  }

  private NDArray _inputGradient;
  private double backpropPruning = 0.;
  private GradientDescentAccumulator deltaBuffer;
  private boolean frozen = false;
  private final int[] outputDims;
  private boolean verbose = false;
  
  public final NDArray weights;
  
  private DeltaFlushBuffer writer;
  
  protected DenseSynapseLayer() {
    super();
    this.outputDims = null;
    this.weights = null;
  }
  
  public DenseSynapseLayer(final int inputs, final int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputDims));
    this.writer = new DeltaFlushBuffer(this.weights);
    this.deltaBuffer = new GradientDescentAccumulator(this.writer);
  }
  
  public DenseSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray output = new NDArray(this.outputDims);
    final NDArray inputGradient = null != this._inputGradient ? null : new NDArray(input.dim(), output.dim());
    final NDArray weightGradient = this.frozen ? null : new NDArray(this.weights.dim(), output.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(o -> {
        final double a = this.weights.get(i, o);
        final double b = input.getData()[i];
        if (null != inputGradient) {
          inputGradient.add(new int[] { i, o }, a);
        }
        if (null != weightGradient) {
          weightGradient.add(new int[] { this.weights.index(i, o), o }, b);
        }
        final double value = b * a;
        if (Double.isFinite(value)) {
          output.add(o, value);
        }
      });
    });
    if (null != inputGradient) {
      this._inputGradient = inputGradient;
    }
    if (isVerbose()) {
      DenseSynapseLayer.log.debug(String.format("Feed forward: %s * %s => %s", inObj.data, this.weights, output));
    }
    return new DenseSynapseResult(output, this._inputGradient, weightGradient, inObj);
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
  public double getRate() {
    return this.writer.getRate();
  }
  
  @Override
  public boolean isFrozen() {
    return this.frozen;
  }
  
  private boolean isVerbose() {
    return this.verbose;
  }
  
  public DenseSynapseLayer setBackpropPruning(final double backpropPruning) {
    this.backpropPruning = backpropPruning;
    return this;
  }
  
  @Override
  public void setRate(final double rate) {
    this.writer.setRate(rate);
  }
  
  public DenseSynapseLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public DenseSynapseLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }

  public DenseSynapseLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }

  public DenseSynapseLayer thaw() {
    return freeze(false);
  }

  @Override
  public String toString() {
    return "DenseSynapseLayer [weights=" + this.weights + "]";
  }

  @Override
  public void write(final double factor) {
    this._inputGradient = null;
    this.writer.write(factor);
  }
}
