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
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Bias meta layer.
 */
@SuppressWarnings("serial")
public class BiasMetaLayer extends NNLayer {
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasMetaLayer.class);
  
  /**
   * Instantiates a new Bias meta layer.
   */
  public BiasMetaLayer() {
  }
  
  /**
   * Instantiates a new Bias meta layer.
   *
   * @param id the id
   */
  protected BiasMetaLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json bias meta layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the bias meta layer
   */
  public static BiasMetaLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new BiasMetaLayer(json);
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    final Tensor[] tensors = IntStream.range(0, itemCnt)
                                      .parallel()
                                      .mapToObj(dataIndex -> inObj[0].getData().get(dataIndex).mapIndex((v, c) -> v + inObj[1].getData().get(0).get(c)))
                                      .toArray(i -> new Tensor[i]);
    Tensor tensor0 = tensors[0];
    tensor0.addRef();
        Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    return new NNResult(TensorArray.wrap(tensors), (final DeltaSet<NNLayer> buffer, final TensorList data) -> {
      if (inObj[0].isAlive()) {
        TensorArray delta = TensorArray.wrap(data.stream().map(t -> t.mapParallel(v -> v)).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, delta);
        delta.freeRef();
      }
      if (inObj[1].isAlive()) {
        final ToDoubleFunction<Coordinate> f = (c) -> {
          return IntStream.range(0, itemCnt).mapToDouble(i -> data.get(i).get(c)).sum();
        };
        final Tensor passback = tensor0.mapCoords(f);
       TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, inObj[1].getData().length())
                                                            .mapToObj(i -> i == 0 ? passback : passback.map(v -> 0)).toArray(i -> new Tensor[i]));
        inObj[1].accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
    
      @Override
      protected void _free() {
        tensor0.freeRef();
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || inObj[1].isAlive();
      }
      
    };
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
