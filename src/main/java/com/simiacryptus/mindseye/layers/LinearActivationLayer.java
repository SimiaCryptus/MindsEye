package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext;
import com.simiacryptus.mindseye.util.Util;

public class LinearActivationLayer extends NNLayer {
  private final class DenseSynapseResult extends NNResult {
    private final NNResult inObj;

    private DenseSynapseResult(final NDArray data, final NNResult inObj) {
      super(data);
      this.inObj = inObj;
    }

    @Override
    public void feedback(final LogNDArray delta, final DeltaBuffer buffer) {
      if (isVerbose()) {
        LinearActivationLayer.log.debug(String.format("Feed back: %s", this.data));
      }
      final LogNumber[] deltaData = delta.getData();
      
      if (!isFrozen()) {
        final double[] inputData = this.inObj.data.getData();
        final LogNDArray weightDelta = new LogNDArray(LinearActivationLayer.this.weights.getDims());
        for (int i = 0; i < deltaData.length; i++) {
          weightDelta.add(0, deltaData[i].multiply(inputData[i]));
        }
        buffer.get(LinearActivationLayer.this, LinearActivationLayer.this.weights).feed(weightDelta.exp().getData());
      }
      if (this.inObj.isAlive()) {
        final DoubleMatrix matrix = LinearActivationLayer.this.weights.asRowMatrix();
        int[] dims = this.inObj.data.getDims();
        final LogNDArray passback = new LogNDArray(dims);
        for (int i = 0; i < passback.dim(); i++) {
          passback.set(i, deltaData[i].multiply(matrix.get(0, 0)));
        }
        this.inObj.feedback(passback, buffer);
        if (isVerbose()) {
          LinearActivationLayer.log.debug(String.format("Feed back @ %s=>%s: %s => %s", this.inObj.data, DenseSynapseResult.this.data, delta, passback));
        }
      } else {
        if (isVerbose()) {
          LinearActivationLayer.log.debug(String.format("Feed back via @ %s=>%s: %s => null", this.inObj.data, DenseSynapseResult.this.data, delta));
        }
      }
    }

    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }
    
  }

  private static final Logger log = LoggerFactory.getLogger(LinearActivationLayer.class);

  private boolean frozen = false;
  private boolean verbose = false;
  public final NDArray weights;

  public LinearActivationLayer() {
    super();
    this.weights = new NDArray(1);
    this.weights.set(0, 1.);
  }

  public LinearActivationLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }

  @Override
  public NNResult eval(EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = new NDArray(input.getDims());
    IntStream.range(0, input.dim()).forEach(i -> {
      final double a = this.weights.get(0);
      final double b = input.getData()[i];
      final double value = b * a;
      if (Double.isFinite(value)) {
        output.add(i, value);
      }
    });
    if (isVerbose()) {
      LinearActivationLayer.log.debug(String.format("Feed forward: %s * %s => %s", inObj[0].data, this.weights, output));
    }
    return new DenseSynapseResult(output, inObj[0]);
  }

  public LinearActivationLayer freeze() {
    return freeze(true);
  }

  public LinearActivationLayer freeze(final boolean b) {
    this.frozen = b;
    return this;
  }

  protected double getMobility() {
    return 1;
  }

  public boolean isFrozen() {
    return this.frozen;
  }

  private boolean isVerbose() {
    return this.verbose;
  }
  
  public LinearActivationLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
  
  public LinearActivationLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }
  
  public LinearActivationLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }
  
  public LinearActivationLayer thaw() {
    return freeze(false);
  }
  
  @Override
  public JsonObject getJson() {
    JsonObject json = super.getJson();
    json.addProperty("weights", this.weights.toString());
    return json;
  }
  
}
