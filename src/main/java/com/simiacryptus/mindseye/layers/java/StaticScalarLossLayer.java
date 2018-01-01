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
 * The type Static scalar loss layer.
 */
@SuppressWarnings("serial")
public class StaticScalarLossLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StaticScalarLossLayer.class);
  private double target = 0.0;
  
  /**
   * Instantiates a new Static scalar loss layer.
   */
  public StaticScalarLossLayer() {
  }
  
  
  /**
   * Instantiates a new Static scalar loss layer.
   *
   * @param id the id
   */
  protected StaticScalarLossLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json static scalar loss layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the static scalar loss layer
   */
  public static StaticScalarLossLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new StaticScalarLossLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    if (1 != inObj.length) throw new IllegalArgumentException();
    //if (inObj[0].getData().length() != 1) throw new IllegalArgumentException();
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
    final Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
      final Tensor a = inObj[0].getData().get(dataIndex);
      final double diff = Math.abs(a.get(0) - getTarget());
      return new Tensor(new double[]{diff}, 1);
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
  
      @Override
      public void finalize() {
        Arrays.stream(inObj).forEach(NNResult::finalize);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        assert data.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        if (inObj[0].isAlive()) {
          final Tensor[] passbackA = IntStream.range(0, inObj[0].getData().length()).parallel().mapToObj(dataIndex -> {
            final Tensor a = inObj[0].getData().get(dataIndex);
            final double deriv = data.get(dataIndex).get(0) * (a.get(0) - getTarget() < 0 ? -1 : 1);
            return new Tensor(new double[]{deriv}, 1);
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
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
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
  public StaticScalarLossLayer setTarget(final double target) {
    this.target = target;
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
