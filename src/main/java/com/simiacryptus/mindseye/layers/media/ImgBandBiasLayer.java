/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.media;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.DeltaBuffer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

public class ImgBandBiasLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(bias));
    return json;
  }
  
  public static ImgBandBiasLayer fromJson(JsonObject json) {
    return new ImgBandBiasLayer(json);
  }
  protected ImgBandBiasLayer(JsonObject json) {
    super(json);
    this.bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandBiasLayer.class);
  
  
  public final double[] bias;
  
  protected ImgBandBiasLayer() {
    super();
    this.bias = null;
  }
  
  public ImgBandBiasLayer(final int bands) {
    super();
    this.bias = new double[bands];
  }
  
  public double[] add(final double[] input) {
    assert(null != input);
    assert(null != bias);
    final double[] array = new double[input.length];
    int size = input.length / bias.length;
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + this.bias[i/size];
    }
    return array;
  }
  
  public ImgBandBiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.bias);
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    return eval(inObj[0]);
  }
  
  public NNResult eval(NNResult input) {
    Tensor[] outputA = Arrays.stream(input.data).parallel()
                           .map(r -> new Tensor(r.getDims(), add(r.getData())))
                           .toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (!isFrozen()) {
          DeltaBuffer deltaBuffer = buffer.get(ImgBandBiasLayer.this, ImgBandBiasLayer.this.bias);
          Arrays.stream(data).parallel().forEach(d -> {
            final double[] array = Tensor.obtain(bias.length);
            double[] signal = d.getData();
            int size = signal.length / bias.length;
            for (int i = 0; i < signal.length; i++) {
              array[i/size] += signal[i];
            }
            deltaBuffer.accumulate(array);
            Tensor.recycle(array);
          });
        }
        if (input.isAlive()) {
          input.accumulate(buffer, data);
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  public NNLayer set(final double[] ds) {
    for (int i = 0; i < ds.length; i++) {
      this.bias[i] = ds[i];
    }
    return this;
  }
  
  public ImgBandBiasLayer setWeights(final IntToDoubleFunction f) {
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
