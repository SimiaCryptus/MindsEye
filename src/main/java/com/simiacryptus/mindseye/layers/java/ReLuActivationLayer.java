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
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

/**
 * Rectified Linear Unit. y=(x&lt;0)?0:x
 */
@SuppressWarnings("serial")
public class ReLuActivationLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ReLuActivationLayer.class);
  private final Tensor weights;
  
  /**
   * Instantiates a new Re lu activation layer.
   */
  public ReLuActivationLayer() {
    super();
    weights = new Tensor(1);
    weights.set(0, 1.);
    setFrozen(true);
  }
  
  
  /**
   * Instantiates a new Re lu activation layer.
   *
   * @param json      the json
   * @param resources the resources
   */
  protected ReLuActivationLayer(final JsonObject json, Map<String, byte[]> resources) {
    super(json);
    weights = Tensor.fromJson(json.get("weights"), resources);
  }
  
  /**
   * From json re lu activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the re lu activation layer
   */
  public static ReLuActivationLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new ReLuActivationLayer(json, rs);
  }
  
  /**
   * Add weights re lu activation layer.
   *
   * @param f the f
   * @return the re lu activation layer
   */
  public ReLuActivationLayer addWeights(final DoubleSupplier f) {
    Util.add(f, weights.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
    final NNResult input = inObj[0];
    final TensorList indata = input.getData();
    indata.addRef();
    final int itemCnt = indata.length();
    final Tensor[] output = IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
      final Tensor tensor = indata.get(dataIndex).multiply(weights.get(0));
      final double[] outputData = tensor.getData();
      for (int i = 0; i < outputData.length; i++) {
        if (outputData[i] < 0) {
          outputData[i] = 0;
        }
      }
      return tensor;
    }).toArray(i -> new Tensor[i]);
    assert Arrays.stream(output).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    return new NNResult(TensorArray.wrap(output), (final DeltaSet<NNLayer> buffer, final TensorList delta) -> {
      assert delta.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
      if (!isFrozen()) {
        IntStream.range(0, delta.length()).parallel().forEach(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final double[] inputData = indata.get(dataIndex).getData();
          final Tensor weightDelta = new Tensor(weights.getDimensions());
          final double[] weightDeltaData = weightDelta.getData();
          for (int i = 0; i < deltaData.length; i++) {
            weightDeltaData[0] += inputData[i] < 0 ? 0 : deltaData[i] * inputData[i];
          }
          buffer.get(ReLuActivationLayer.this, weights.getData()).addInPlace(weightDeltaData);
        });
      }
      if (input.isAlive()) {
        final double weight = weights.getData()[0];
        TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).parallel().mapToObj(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final double[] inputData = indata.get(dataIndex).getData();
          final int[] dims = indata.get(dataIndex).getDimensions();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, inputData[i] < 0 ? 0 : deltaData[i] * weight);
          }
          return passback;
        }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
      indata.freeRef();
    }) {
    
      @Override
      protected void _free() {
        input.freeRef();
      }
    
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    
    };
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("weights", weights.toJson(resources, dataSerializer));
    return json;
  }
  
  /**
   * Gets mobility.
   *
   * @return the mobility
   */
  protected double getMobility() {
    return 1;
  }
  
  /**
   * Sets weight.
   *
   * @param data the data
   * @return the weight
   */
  public ReLuActivationLayer setWeight(final double data) {
    weights.set(0, data);
    return this;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  public ReLuActivationLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(weights.getData());
  }
  
}
