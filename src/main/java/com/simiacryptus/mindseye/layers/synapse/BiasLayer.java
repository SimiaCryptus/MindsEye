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
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

/**
 * The type Bias layer.
 */
public class BiasLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(bias));
    return json;
  }
  
  /**
   * From json bias layer.
   *
   * @param json the json
   * @return the bias layer
   */
  public static BiasLayer fromJson(JsonObject json) {
    return new BiasLayer(json);
  }

  /**
   * Instantiates a new Bias layer.
   *
   * @param json the json
   */
  protected BiasLayer(JsonObject json) {
    super(json);
    this.bias = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  
  
  /**
   * The Bias.
   */
  public final double[] bias;
  
  /**
   * Instantiates a new Bias layer.
   */
  protected BiasLayer() {
    super();
    this.bias = null;
  }
  
  /**
   * Instantiates a new Bias layer.
   *
   * @param outputDims the output dims
   */
  public BiasLayer(final int... outputDims) {
    this.bias = new double[Tensor.dim(outputDims)];
  }
  
  /**
   * Add double [ ].
   *
   * @param input the input
   * @return the double [ ]
   */
  public double[] add(final double[] input) {
    final double[] array = Tensor.obtain(input.length);
    for (int i = 0; i < array.length; i++) {
      array[i] = input[i] + this.bias[i];
    }
    return array;
  }
  
  /**
   * Add weights bias layer.
   *
   * @param f the f
   * @return the bias layer
   */
  public BiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.bias);
    return this;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    TensorList input;
    if(0==inObj.length) {
      input = new TensorArray();
    } else {
      input = inObj[0].data;
    }
    Tensor[] outputA = input.stream().parallel()
                           .map(r -> new Tensor(r.getDimensions(), add(r.getData())))
                           .toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (!isFrozen()) {
          DeltaBuffer deltaBuffer = buffer.get(BiasLayer.this, BiasLayer.this.bias);
          data.stream().parallel().forEach(d -> deltaBuffer.accumulate(d.getData()));
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
  
  
  public BiasLayer setWeightsLog(final double value) {
    for(int i=0;i<this.bias.length;i++) this.bias[i] = (FastRandom.random()-0.5)*Math.pow(10,value);
    return this;
  }
  
  /**
   * Set nn layer.
   *
   * @param ds the ds
   * @return the nn layer
   */
  public NNLayer set(final double[] ds) {
    for (int i = 0; i < ds.length; i++) {
      this.bias[i] = ds[i];
    }
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
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
