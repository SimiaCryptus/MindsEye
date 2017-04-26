package com.simiacryptus.mindseye.net.basic;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.delta.DeltaBuffer;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class BiasLayer extends NNLayer<BiasLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);

  private static final long serialVersionUID = 1022169631431441049L;

  public final double[] bias;

  protected BiasLayer() {
    super();
    this.bias = null;
  }

  public BiasLayer(final int... outputDims) {
    this.bias = new double[Tensor.dim(outputDims)];
  }

  public double[] add(final double[] input) {
    final double[] array = new double[input.length];
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + this.bias[i];
    }
    return array;
  }

  public BiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.bias);
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    Tensor[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
      final Tensor r = inObj[0].data[dataIndex];
      return new Tensor(r.getDims(), add(r.getData()));
    }).toArray(i->new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (!isFrozen()) {
          java.util.stream.IntStream.range(0, data.length).forEach(dataIndex->{
            Tensor tensor = data[dataIndex];
            double[] data2 = tensor.getData();
            DeltaBuffer deltaBuffer = buffer.get(BiasLayer.this, BiasLayer.this.bias);
            deltaBuffer.feed(data2);
          });
        }
        if (inObj[0].isAlive()) {
          inObj[0].accumulate(buffer, data);
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    };
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
