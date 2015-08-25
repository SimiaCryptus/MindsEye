package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaFlushBuffer;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;

public class DenseSynapseLayer extends NNLayer {
  private final class DenseSynapseResult extends NNResult {
    private final NNResult inObj;
    
    private DenseSynapseResult(final NDArray data, final NNResult inObj) {
      super(data);
      this.inObj = inObj;
    }
    
    @Override
    public void feedback(final LogNDArray delta) {
      if (isVerbose()) {
        log.debug(String.format("Feed back: %s", data));
      }
      LogNumber[] deltaData = delta.getData();
      double[] inputData = inObj.data.getData();

      LogNDArray weightDelta = new LogNDArray(weights.getDims());
      for(int i=0;i<weightDelta.getDims()[0];i++){
        for(int j=0;j<weightDelta.getDims()[1];j++){
          weightDelta.set(new int[]{i,j}, deltaData[j].multiply(inputData[i]));
        }
      }
      writer.feed(weightDelta.exp().getData());
      if (this.inObj.isAlive()) {
        DoubleMatrix matrix = weights.asMatrix();
        //DoubleMatrix deltaV = new DoubleMatrix(deltaData.length, 1, deltaData);
        //final double[] inverted = matrix.mmul(deltaV).data;
        LogNDArray passback = new LogNDArray(this.inObj.data.getDims());
        for(int i=0;i<matrix.columns;i++){
          for(int j=0;j<matrix.rows;j++){
            passback.add(i, deltaData[j].multiply(matrix.get(j, i)));
          }
        }
        this.inObj.feedback(passback);
        if (isVerbose()) {
          DenseSynapseLayer.log.debug(String.format("Feed back @ %s=>%s: %s => %s", inObj.data, DenseSynapseResult.this.data, delta, passback));
        }
      } else {
        //this.inObj.feedback(null);
        if (isVerbose()) {
          DenseSynapseLayer.log.debug(String.format("Feed back via @ %s=>%s: %s => null", inObj.data, DenseSynapseResult.this.data, delta));
        }
      }
    }
    
    @Override
    public boolean isAlive() {
      return inObj.isAlive() || !isFrozen();
    }

  }
  
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);
  
  public static int[] transpose(final int[] dims2) {
    final int[] dims = new int[] { dims2[1], dims2[0] };
    return dims;
  }

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
  }
  
  public DenseSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final NDArray output = new NDArray(this.outputDims);
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(o -> {
        final double a = this.weights.get(i, o);
        final double b = input.getData()[i];
        final double value = b * a;
        if (Double.isFinite(value)) {
          output.add(o, value);
        }
      });
    });
    if (isVerbose()) {
      DenseSynapseLayer.log.debug(String.format("Feed forward: %s * %s => %s", inObj.data, this.weights, output));
    }
    return new DenseSynapseResult(output, inObj);
  }
  
  public DenseSynapseLayer freeze() {
    return freeze(true);
  }
  
  public DenseSynapseLayer freeze(final boolean b) {
    this.frozen = b;
    return this;
  }
  
  public double getRate() {
    return this.writer.getRate();
  }
  
  public boolean isFrozen() {
    return this.frozen;
  }
  
  private boolean isVerbose() {
    return this.verbose;
  }
  
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

  protected DeltaTransaction newVector(double fraction,long mask) {
    if (isFrozen()) return null;
    return new DeltaTransaction() {
      
      @Override
      public void write(double factor) {
        DenseSynapseLayer.this.writer.write(factor, fraction, mask);
      }
      
      @Override
      public void setRate(double rate) {
        DenseSynapseLayer.this.setRate(rate);
      }
      
      @Override
      public boolean isFrozen() {
        return DenseSynapseLayer.this.isFrozen();
      }
      
      @Override
      public double getRate() {
        return DenseSynapseLayer.this.getRate();
      }
    };
  }

}
