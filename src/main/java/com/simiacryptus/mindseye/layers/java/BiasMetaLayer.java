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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * The type Bias meta layer.
 */
@SuppressWarnings("serial")
public class BiasMetaLayer extends LayerBase {


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
  protected BiasMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * From json bias meta layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the bias meta layer
   */
  public static BiasMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BiasMetaLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final int itemCnt = inObj[0].getData().length();
    Tensor tensor1 = inObj[1].getData().get(0);
    final Tensor[] tensors = IntStream.range(0, itemCnt)
        .parallel()
        .mapToObj(dataIndex -> {
          Tensor tensor = inObj[0].getData().get(dataIndex);
          Tensor mapIndex = tensor.mapIndex((v, c) -> {
            return v + tensor1.get(c);
          });
          tensor.freeRef();
          return mapIndex;
        })
        .toArray(i -> new Tensor[i]);
    tensor1.freeRef();
    Tensor tensor0 = tensors[0];
    tensor0.addRef();
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    return new Result(TensorArray.wrap(tensors), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
      if (inObj[0].isAlive()) {
        data.addRef();
        inObj[0].accumulate(buffer, data);
      }
      if (inObj[1].isAlive()) {
        @Nonnull final ToDoubleFunction<Coordinate> f = (c) -> {
          return IntStream.range(0, itemCnt).mapToDouble(i -> {
            Tensor tensor = data.get(i);
            double v = tensor.get(c);
            tensor.freeRef();
            return v;
          }).sum();
        };
        @Nullable final Tensor passback = tensor0.mapCoords(f);
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, inObj[1].getData().length())
            .mapToObj(i -> {
              if (i == 0) return passback;
              else {
                @Nullable Tensor map = passback.map(v -> 0);
                passback.freeRef();
                return map;
              }
            }).toArray(i -> new Tensor[i]));
        inObj[1].accumulate(buffer, tensorArray);
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

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
