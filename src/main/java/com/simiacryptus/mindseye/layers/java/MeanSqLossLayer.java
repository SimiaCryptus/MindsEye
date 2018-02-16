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

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
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
  protected MeanSqLossLayer(@javax.annotation.Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json mean sq loss layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the mean sq loss layer
   */
  public static MeanSqLossLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new MeanSqLossLayer(json);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    if (2 != inObj.length) throw new IllegalArgumentException();
    final int leftLength = inObj[0].getData().length();
    final int rightLength = inObj[1].getData().length();
    Arrays.stream(inObj).forEach(ReferenceCountingBase::addRef);
    if (leftLength != rightLength && leftLength != 1 && rightLength != 1) {
      throw new IllegalArgumentException(leftLength + " != " + rightLength);
    }
    @javax.annotation.Nonnull final Tensor diffs[] = new Tensor[leftLength];
    return new NNResult(TensorArray.wrap(IntStream.range(0, leftLength).mapToObj(dataIndex -> {
      @javax.annotation.Nullable final Tensor a = inObj[0].getData().get(1 == leftLength ? 0 : dataIndex);
      @javax.annotation.Nullable final Tensor b = inObj[1].getData().get(1 == rightLength ? 0 : dataIndex);
      if (a.dim() != b.dim()) {
        throw new IllegalArgumentException(String.format("%s != %s", Arrays.toString(a.getDimensions()), Arrays.toString(b.getDimensions())));
      }
      @javax.annotation.Nonnull final Tensor r = a.minus(b);
      a.freeRef();
      b.freeRef();
      diffs[dataIndex] = r;
      @javax.annotation.Nonnull Tensor statsTensor = new Tensor(new double[]{r.sumSq() / r.dim()}, 1);
      return statsTensor;
    }).toArray(i -> new Tensor[i])), (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      if (inObj[0].isAlive()) {
        Stream<Tensor> tensorStream = IntStream.range(0, data.length()).parallel().mapToObj(dataIndex -> {
          @Nullable Tensor tensor = data.get(dataIndex);
          Tensor diff = diffs[dataIndex];
          @Nullable Tensor scale = diff.scale(tensor.get(0) * 2.0 / diff.dim());
          tensor.freeRef();
          return scale;
        }).collect(Collectors.toList()).stream();
        if (1 == leftLength) {
          tensorStream = Stream.of(tensorStream.reduce((a, b) -> {
            @javax.annotation.Nullable Tensor c = a.addAndFree(b);
            b.freeRef();
            return c;
          }).get());
        }
        @javax.annotation.Nonnull final TensorList array = TensorArray.wrap(tensorStream.toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, array);
        array.freeRef();
      }
      if (inObj[1].isAlive()) {
        Stream<Tensor> tensorStream = IntStream.range(0, data.length()).parallel().mapToObj(dataIndex -> {
          @Nullable Tensor tensor = data.get(dataIndex);
          @Nullable Tensor scale = diffs[dataIndex].scale(tensor.get(0) * 2.0 / diffs[dataIndex].dim());
          tensor.freeRef();
          return scale;
        }).collect(Collectors.toList()).stream();
        if (1 == rightLength) {
          tensorStream = Stream.of(tensorStream.reduce((a, b) -> {
            @javax.annotation.Nullable Tensor c = a.addAndFree(b);
            b.freeRef();
            return c;
          }).get());
        }
        @javax.annotation.Nonnull final TensorList array = TensorArray.wrap(tensorStream.map(x -> {
          @javax.annotation.Nullable Tensor scale = x.scale(-1);
          x.freeRef();
          return scale;
        }).toArray(i -> new Tensor[i]));
        inObj[1].accumulate(buffer, array);
        array.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(ReferenceCountingBase::freeRef);
        Arrays.stream(diffs).forEach(ReferenceCountingBase::freeRef);
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || inObj[1].isAlive();
      }
      
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
