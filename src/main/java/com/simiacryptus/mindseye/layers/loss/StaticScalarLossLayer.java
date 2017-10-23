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

package com.simiacryptus.mindseye.layers.loss;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Static scalar loss layer.
 */
public class StaticScalarLossLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
  /**
   * From json static scalar loss layer.
   *
   * @param json the json
   * @return the static scalar loss layer
   */
  public static StaticScalarLossLayer fromJson(JsonObject json) {
    return new StaticScalarLossLayer(json);
  }
  
  /**
   * Instantiates a new Static scalar loss layer.
   *
   * @param id the id
   */
  protected StaticScalarLossLayer(JsonObject id) {
    super(id);
  }
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StaticScalarLossLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = 7589211270512485408L;
  
  private double target = 0.0;
  
  /**
   * Instantiates a new Static scalar loss layer.
   */
  public StaticScalarLossLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    if (1 != inObj.length) throw new IllegalArgumentException();
    if (inObj[0].getData().length() != 1) throw new IllegalArgumentException();
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
    Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
      final Tensor a = inObj[0].getData().get(dataIndex);
      final double diff = Math.abs(a.get(0) - getTarget());
      return new Tensor(new int[]{1}, new double[]{diff});
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        assert data.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
            final Tensor a = inObj[0].getData().get(dataIndex);
            final double deriv = data.get(dataIndex).get(0) * ((a.get(0) - getTarget()) < 0 ? -1 : 1);
            return new Tensor(new int[]{1}, new double[]{deriv});
          }).toArray(i -> new Tensor[i]);
          inObj[0].accumulate(buffer, new TensorArray(passbackA));
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  /**
   * Gets target.
   *
   * @return the target
   */
  public double getTarget() {
    return target;
  }
  
  /**
   * Sets target.
   *
   * @param target the target
   * @return the target
   */
  public StaticScalarLossLayer setTarget(double target) {
    this.target = target;
    return this;
  }
}
