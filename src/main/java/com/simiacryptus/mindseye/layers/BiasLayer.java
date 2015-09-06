package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext;
import com.simiacryptus.mindseye.util.Util;

public class BiasLayer extends NNLayer {

  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);

  public final double[] bias;
  private boolean frozen = false;
  private boolean verbose = false;

  protected BiasLayer() {
    super();
    this.bias = null;
  }

  public BiasLayer(final int[] outputDims) {
    this.bias = new double[NDArray.dim(outputDims)];
  }

  public BiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.bias);
    return this;
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray translated = inObj[0].data.map((v, i) -> {
      return v + this.bias[i.index];
    });
    if (isVerbose()) {
      BiasLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, translated));
    }
    return new NNResult(translated) {
      @Override
      public void feedback(final LogNDArray data, final DeltaBuffer buffer) {
        if (isVerbose()) {
          BiasLayer.log.debug(String.format("Feed back: %s", data));
        }
        if (!isFrozen()) {
          buffer.get(BiasLayer.this, BiasLayer.this.bias).feed(data.getData());
        }
        if (inObj[0].isAlive()) {
          inObj[0].feedback(data, buffer);
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    };
  }

  public BiasLayer freeze() {
    return setFrozen(true);
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("bias", Arrays.toString(this.bias));
    return json;
  }

  public boolean isFrozen() {
    return this.frozen;
  }

  @Override
  public boolean isVerbose() {
    return this.verbose;
  }

  public NNLayer set(final double[] ds) {
    for (int i = 0; i < ds.length; i++) {
      this.bias[i] = ds[i];
    }
    return this;
  }

  public BiasLayer setFrozen(final boolean frozen) {
    this.frozen = frozen;
    return this;
  }

  public BiasLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.bias);
  }

}
