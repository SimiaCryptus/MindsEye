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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The type Static scalar loss layer.
 */
@SuppressWarnings("serial")
public class StaticScalarLossLayer extends LayerBase {
  
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
  protected StaticScalarLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json static scalar loss layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the static scalar loss layer
   */
  public static StaticScalarLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new StaticScalarLossLayer(json);
  }
  
  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (1 != inObj.length) throw new IllegalArgumentException();
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    //if (inObj[0].getData().length() != 1) throw new IllegalArgumentException();
    final Result in0 = inObj[0];
    TensorList indata = in0.getData();
    indata.addRef();
    return new Result(TensorArray.wrap(IntStream.range(0, indata.length()).parallel().mapToObj(dataIndex -> {
      @Nullable final Tensor a = indata.get(dataIndex);
      final double diff = Math.abs(a.get(0) - getTarget());
      a.freeRef();
      return new Tensor(new double[]{diff}, 1);
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
      if (in0.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, data.length()).parallel().mapToObj(dataIndex -> {
          @Nullable final Tensor a = indata.get(dataIndex);
          Tensor tensor = data.get(dataIndex);
          final double deriv = tensor.get(0) * (a.get(0) - getTarget() < 0 ? -1 : 1);
          tensor.freeRef();
          a.freeRef();
          return new Tensor(new double[]{deriv}, 1);
        }).toArray(i -> new Tensor[i]));
        in0.accumulate(buffer, tensorArray);
      }
    }) {
      
      @Override
      protected void _free() {
        indata.freeRef();
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      
      @Override
      public boolean isAlive() {
        return in0.isAlive();
      }
      
    };
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
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
  @Nonnull
  public StaticScalarLossLayer setTarget(final double target) {
    this.target = target;
    return this;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
