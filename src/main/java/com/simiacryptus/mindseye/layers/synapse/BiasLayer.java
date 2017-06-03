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

package com.simiacryptus.mindseye.layers.synapse;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.DeltaBuffer;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
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

public class BiasLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(bias));
    return json;
  }
  
  public static BiasLayer fromJson(JsonObject json) {
    return new BiasLayer(json);
  }
  protected BiasLayer(JsonObject json) {
    super(UUID.fromString(json.get("id").getAsString()));
    this.bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  
  
  public final double[] bias;
  
  protected BiasLayer() {
    super();
    this.bias = null;
  }
  
  public BiasLayer(final int... outputDims) {
    this.bias = new double[Tensor.dim(outputDims)];
  }
  
  public double[] add(final double[] input) {
    final double[] array = Tensor.obtain(input.length);
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
    Tensor[] input;
    if(0==inObj.length) {
      input = new Tensor[]{};
    } else {
      input = inObj[0].data;
    }
    Tensor[] outputA = Arrays.stream(input).parallel()
                           .map(r -> new Tensor(r.getDims(), add(r.getData())))
                           .toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (!isFrozen()) {
          DeltaBuffer deltaBuffer = buffer.get(BiasLayer.this, BiasLayer.this.bias);
          Arrays.stream(data).parallel().forEach(d -> deltaBuffer.accumulate(d.getData()));
        }
        if (0 < inObj.length && inObj[0].isAlive()) {
          inObj[0].accumulate(buffer, data);
        }
      }
      
      @Override
      public boolean isAlive() {
        return (0 < inObj.length && inObj[0].isAlive()) || !isFrozen();
      }
    };
  }
  
  
  public NNLayer set(final double[] ds) {
    for (int i = 0; i < ds.length; i++) {
      this.bias[i] = ds[i];
    }
    return this;
  }
  
  public BiasLayer setWeights(final IntToDoubleFunction f) {
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
