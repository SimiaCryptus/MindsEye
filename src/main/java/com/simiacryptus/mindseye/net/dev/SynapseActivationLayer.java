package com.simiacryptus.mindseye.net.dev;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.NNLayer;

public class SynapseActivationLayer extends NNLayer<SynapseActivationLayer> {
  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final NDArray data, final NNResult inObj) {
      super(data);
      this.inObj = inObj;
    }

    @Override
    public void feedback(final NDArray delta, final DeltaSet buffer) {
      final double[] deltaData = delta.getData();

      if (!isFrozen()) {
        final double[] inputData = this.inObj.data.getData();
        final NDArray weightDelta = new NDArray(SynapseActivationLayer.this.weights.getDims());
        for (int i = 0; i < weightDelta.getDims()[0]; i++) {
          weightDelta.set(i, deltaData[i] * inputData[i]);
        }
        buffer.get(SynapseActivationLayer.this, SynapseActivationLayer.this.weights).feed(weightDelta.getData());
      }
      if (this.inObj.isAlive()) {
        final DoubleMatrix matrix = SynapseActivationLayer.this.weights.asRowMatrix();
        final NDArray passback = new NDArray(this.inObj.data.getDims());
        for (int i = 0; i < matrix.columns; i++) {
          passback.set(i, deltaData[i] * matrix.get(i, 0));
        }
        this.inObj.feedback(passback, buffer);
      }
    }

    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }

  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SynapseActivationLayer.class);

  /**
   * 
   */
  private static final long serialVersionUID = -3526706430472468300L;

  public final NDArray weights;

  protected SynapseActivationLayer() {
    super();
    this.weights = null;
  }

  public SynapseActivationLayer(final int inputs) {
    this.weights = new NDArray(inputs);
  }

  public SynapseActivationLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = new NDArray(input.getDims());
    IntStream.range(0, input.dim()).forEach(i -> {
      final double a = this.weights.get(i);
      final double b = input.getData()[i];
      final double value = b * a;
      if (Double.isFinite(value)) {
        output.add(i, value);
      }
    });
    return new Result(output, inObj[0]);
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("weights", this.weights.toString());
    return json;
  }

  protected double getMobility() {
    return 1;
  }

  public SynapseActivationLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }

  public SynapseActivationLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
