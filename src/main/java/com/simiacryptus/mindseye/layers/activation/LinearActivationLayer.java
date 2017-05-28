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

package com.simiacryptus.mindseye.layers.activation;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

public class LinearActivationLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("weights", weights.getJson());
    return json;
  }
  
  public static LinearActivationLayer fromJson(JsonObject json) {
    return new LinearActivationLayer(json);
  }
  protected LinearActivationLayer(JsonObject json) {
    super(UUID.fromString(json.get("id").getAsString()));
    this.weights = Tensor.fromJson(json.getAsJsonObject("weights"));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(LinearActivationLayer.class);
  private final Tensor weights;
  
  public LinearActivationLayer() {
    super();
    this.weights = new Tensor(1);
    this.weights.set(0, 1.);
  }
  
  public LinearActivationLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].data[dataIndex];
      final double a = this.weights.get(0);
      final Tensor output = input.multiply(a);
      return output;
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  
  protected double getMobility() {
    return 1;
  }
  
  public LinearActivationLayer setWeight(final double data) {
    this.weights.set(0, data);
    return this;
  }
  
  public LinearActivationLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.weights.getData());
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    
    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
      
      if (!isFrozen()) {
        IntStream.range(0, delta.length).forEach(dataIndex -> {
          final double[] deltaData = delta[dataIndex].getData();
          final double[] inputData = this.inObj.data[dataIndex].getData();
          final Tensor weightDelta = new Tensor(LinearActivationLayer.this.weights.getDims());
          for (int i = 0; i < deltaData.length; i++) {
            weightDelta.add(0, deltaData[i] * inputData[i]);
          }
          buffer.get(LinearActivationLayer.this, LinearActivationLayer.this.weights).accumulate(weightDelta.getData());
        });
      }
      if (this.inObj.isAlive()) {
        Tensor[] passbackA = IntStream.range(0, delta.length).mapToObj(dataIndex -> {
          final double[] deltaData = delta[dataIndex].getData();
          final int[] dims = this.inObj.data[dataIndex].getDims();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, deltaData[i] * LinearActivationLayer.this.weights.getData()[0]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]);
        this.inObj.accumulate(buffer, passbackA);
      }
    }
    
    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }
    
  }
  
}
