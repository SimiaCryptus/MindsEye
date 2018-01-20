/*
 * Copyright (c) 2018 by Andrew Charneski.
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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * A tunable linear (y=A*x+B) function, whose parameters can participate in learning. Defaults to y=1*x+0, and is NOT
 * frozen by default.
 */
@SuppressWarnings("serial")
public class LinearActivationLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(LinearActivationLayer.class);
  private final Tensor weights;
  
  /**
   * Instantiates a new Linear activation layer.
   */
  public LinearActivationLayer() {
    super();
    weights = new Tensor(2);
    weights.set(0, 1.);
    weights.set(1, 0.);
  }
  
  /**
   * Instantiates a new Linear activation layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected LinearActivationLayer(final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
  }
  
  /**
   * From json linear activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the linear activation layer
   */
  public static LinearActivationLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new LinearActivationLayer(json, rs);
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    final double scale = weights.get(0);
    final double bias = weights.get(1);
    final Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      return input.map(v -> scale * v + bias);
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }
  
  /**
   * Gets bias.
   *
   * @return the bias
   */
  public double getBias() {
    return weights.get(1);
  }
  
  /**
   * Sets bias.
   *
   * @param bias the bias
   * @return the bias
   */
  public LinearActivationLayer setBias(final double bias) {
    weights.set(1, bias);
    return this;
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("weights", weights.toJson(resources, dataSerializer));
    return json;
  }
  
  /**
   * Gets scale.
   *
   * @return the scale
   */
  public double getScale() {
    return weights.get(0);
  }
  
  /**
   * Sets scale.
   *
   * @param scale the scale
   * @return the scale
   */
  public LinearActivationLayer setScale(final double scale) {
    weights.set(0, scale);
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(weights.getData());
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    
    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }
    
    @Override
    public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList delta) {
      if (!isFrozen()) {
        IntStream.range(0, delta.length()).forEach(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final double[] inputData = inObj.getData().get(dataIndex).getData();
          final Tensor weightDelta = new Tensor(weights.getDimensions());
          for (int i = 0; i < deltaData.length; i++) {
            weightDelta.add(0, deltaData[i] * inputData[inputData.length == 1 ? 0 : i]);
            weightDelta.add(1, deltaData[i]);
          }
          buffer.get(LinearActivationLayer.this, weights.getData()).addInPlace(weightDelta.getData());
        });
      }
      if (inObj.isAlive()) {
        final TensorList tensorList = new TensorArray(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final int[] dims = inObj.getData().get(dataIndex).getDimensions();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, deltaData[i] * weights.getData()[0]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj.accumulate(buffer, tensorList);
        tensorList.recycle();
      }
    }
    
    @Override
    public boolean isAlive() {
      return inObj.isAlive() || !isFrozen();
    }
  
    @Override
    public void free() {
      inObj.free();
    }
  
  }
}
