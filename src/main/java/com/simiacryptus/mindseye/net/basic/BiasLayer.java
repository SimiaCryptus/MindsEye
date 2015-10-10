package com.simiacryptus.mindseye.net.basic;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.net.NNLayer;

public class BiasLayer extends NNLayer<BiasLayer> {

  private static final long serialVersionUID = 1022169631431441049L;

  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);

  public final double[] bias;

  protected BiasLayer() {
    super();
    this.bias = null;
  }

  public BiasLayer(final int... outputDims) {
    this.bias = new double[NDArray.dim(outputDims)];
  }

  public BiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.bias);
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray r = inObj[0].data;
    final NDArray translated = new NDArray(r.getDims(), add(r.getData()));
    if (isVerbose()) {
      BiasLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, translated));
    }
    return new NNResult(translated) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
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

  public double[] add(final double[] input) {
    final double[] array = new double[input.length];
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + this.bias[i];
    }
    return array;
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("bias", Arrays.toString(this.bias));
    return json;
  }

  public NNLayer<?> set(final double[] ds) {
    for (int i = 0; i < ds.length; i++) {
      this.bias[i] = ds[i];
    }
    return this;
  }

  public BiasLayer setWeights(final java.util.function.IntToDoubleFunction f) {
    for (int i = 0; i < this.bias.length; i++) {
      this.bias[i] = f.applyAsDouble(i);
    }
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.bias);
  }

}
