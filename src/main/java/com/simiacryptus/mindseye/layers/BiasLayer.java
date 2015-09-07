package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.HashMap;
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

import groovy.lang.Tuple2;

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

  @Override
  public List<Tuple2<Integer, Integer>> permuteInput(final List<Tuple2<Integer, Integer>> permute) {
    final java.util.Map<Integer, Integer> shuffleMap = new HashMap<>();
    java.util.stream.IntStream.range(0, this.bias.length).forEach(i -> shuffleMap.put(i, i));
    ;
    permute.forEach(t -> {
      Integer from = t.getFirst();
      final Integer to = t.getSecond();
      from = shuffleMap.get(from);

      final double temp = this.bias[to];
      this.bias[to] = this.bias[from];
      this.bias[from] = temp;

      for (final int k : shuffleMap.keySet()) {
        int value = shuffleMap.get(k);
        if (value == from) {
          value = to;
        } else if (value == to) {
          value = from;
        }
        shuffleMap.put(k, value);
      }
    });
    return permute;
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteOutput(final List<Tuple2<Integer, Integer>> permute) {
    final java.util.Map<Integer, Integer> shuffleMap = new HashMap<>();
    java.util.stream.IntStream.range(0, this.bias.length).forEach(i -> shuffleMap.put(i, i));
    ;
    permute.forEach(t -> {
      int from = t.getFirst();
      final int to = t.getSecond();
      from = shuffleMap.get(from);

      final double temp = this.bias[to];
      this.bias[to] = this.bias[from];
      this.bias[from] = temp;

      for (final int k : shuffleMap.keySet()) {
        int value = shuffleMap.get(k);
        if (value == from) {
          value = to;
        } else if (value == to) {
          value = from;
        }
        shuffleMap.put(k, value);
      }
    });
    return permute;
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
