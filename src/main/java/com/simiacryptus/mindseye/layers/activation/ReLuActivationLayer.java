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
import com.simiacryptus.mindseye.layers.media.ImgBandBiasLayer;
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

public class ReLuActivationLayer extends NNLayer {
  
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("weights", weights.getJson());
    return json;
  }
  
  public static ReLuActivationLayer fromJson(JsonObject json) {
    return new ReLuActivationLayer(json);
  }
  protected ReLuActivationLayer(JsonObject json) {
    super(UUID.fromString(json.get("id").getAsString()));
    this.weights = Tensor.fromJson(json.getAsJsonObject("weights"));
  }
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ReLuActivationLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = -2105152439043901220L;
  private final Tensor weights;
  
  public ReLuActivationLayer() {
    super();
    this.weights = new Tensor(1);
    this.weights.set(0, 1.);
  }
  
  public ReLuActivationLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    Tensor[] outputA = IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
      final Tensor input = inObj[0].data[dataIndex];
      final double a = this.weights.get(0);
      final Tensor output = input.multiply(a);
      double[] outputData = output.getData();
      for (int i = 0; i < outputData.length; i++) {
        if (outputData[i] < 0) outputData[i] = 0;
      }
      return output;
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  
  protected double getMobility() {
    return 1;
  }
  
  public ReLuActivationLayer setWeight(final double data) {
    this.weights.set(0, data);
    return this;
  }
  
  public ReLuActivationLayer setWeights(final DoubleSupplier f) {
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
        IntStream.range(0, delta.length).parallel().forEach(dataIndex -> {
          final double[] deltaData = delta[dataIndex].getData();
          final double[] inputData = this.inObj.data[dataIndex].getData();
          final Tensor weightDelta = new Tensor(ReLuActivationLayer.this.weights.getDims());
          double[] weightDeltaData = weightDelta.getData();
          for (int i = 0; i < deltaData.length; i++) {
            weightDeltaData[0] = inputData[i] < 0 ? 0 : (deltaData[i] * inputData[i]);
          }
          buffer.get(ReLuActivationLayer.this, ReLuActivationLayer.this.weights).accumulate(weightDeltaData);
        });
      }
      if (this.inObj.isAlive()) {
        double v = ReLuActivationLayer.this.weights.getData()[0];
        Tensor[] passbackA = IntStream.range(0, delta.length).parallel().mapToObj(dataIndex -> {
          final double[] deltaData = delta[dataIndex].getData();
          final double[] inputData = this.inObj.data[dataIndex].getData();
          final int[] dims = this.inObj.data[dataIndex].getDims();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, inputData[i] < 0 ? 0 : (deltaData[i] * v));
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
  
  @Override
  public ReLuActivationLayer freeze() {
    return (ReLuActivationLayer) super.freeze();
  }
}
