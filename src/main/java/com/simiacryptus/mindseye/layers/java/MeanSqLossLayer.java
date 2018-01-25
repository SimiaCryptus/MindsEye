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
import java.util.stream.Stream;

/**
 * An RMS-differencing loss function without the final square root.
 */
@SuppressWarnings("serial")
public class MeanSqLossLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);
  
  /**
   * Instantiates a new Mean sq loss layer.
   */
  public MeanSqLossLayer() {
  }
  
  /**
   * Instantiates a new Mean sq loss layer.
   *
   * @param id the id
   */
  protected MeanSqLossLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json mean sq loss layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the mean sq loss layer
   */
  public static MeanSqLossLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new MeanSqLossLayer(json);
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    if (2 != inObj.length) throw new IllegalArgumentException();
    final int leftLength = inObj[0].getData().length();
    final int rightLength = inObj[1].getData().length();
        Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    if (leftLength != rightLength && leftLength != 1 && rightLength != 1) {
      throw new IllegalArgumentException(leftLength + " != " + rightLength);
    }
    final Tensor diffs[] = new Tensor[leftLength];
    return new NNResult(TensorArray.wrap(IntStream.range(0, leftLength).parallel().mapToObj(dataIndex -> {
      final Tensor a = inObj[0].getData().get(1 == leftLength ? 0 : dataIndex);
      final Tensor b = inObj[1].getData().get(1 == rightLength ? 0 : dataIndex);
      if (a.dim() != b.dim()) {
        throw new IllegalArgumentException(String.format("%s != %s", Arrays.toString(a.getDimensions()), Arrays.toString(b.getDimensions())));
      }
      final Tensor r = a.minus(b);
      diffs[dataIndex] = r;
      return new Tensor(new double[]{r.sumSq() / r.dim()}, 1);
    }).toArray(i -> new Tensor[i])), (final DeltaSet<NNLayer> buffer, final TensorList data) -> {
      if (inObj[0].isAlive() || inObj[1].isAlive()) {
        if (inObj[0].isAlive()) {
          Stream<Tensor> tensorStream = IntStream.range(0, data.length()).parallel().mapToObj(dataIndex -> {
            return diffs[dataIndex].scale(data.get(dataIndex).get(0) * 2.0 / diffs[dataIndex].dim());
          });
          if (1 == leftLength) {
            tensorStream = Stream.of(tensorStream.reduce((a, b) -> a.add(b)).get());
          }
          final TensorList array = TensorArray.wrap(tensorStream.toArray(i -> new Tensor[i]));
          inObj[0].accumulate(buffer, array);
          array.freeRef();
        }
        if (inObj[1].isAlive()) {
          Stream<Tensor> tensorStream = IntStream.range(0, data.length()).parallel().mapToObj(dataIndex -> {
            return diffs[dataIndex].scale(data.get(dataIndex).get(0) * 2.0 / diffs[dataIndex].dim());
          });
          if (1 == rightLength) {
            tensorStream = Stream.of(tensorStream.reduce((a, b) -> a.add(b)).get());
          }
          final TensorList array = TensorArray.wrap(tensorStream.map(x -> x.scale(-1)).toArray(i -> new Tensor[i]));
          inObj[1].accumulate(buffer, array);
          array.freeRef();
        }
      }
    }) {
    
      @Override
      protected void _free() {
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
