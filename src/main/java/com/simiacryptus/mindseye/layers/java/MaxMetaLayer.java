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
import java.util.*;
import java.util.stream.IntStream;

/**
 * The type Max meta key.
 */
@SuppressWarnings("serial")
public class MaxMetaLayer extends LayerBase {


  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxMetaLayer.class);

  /**
   * Instantiates a new Max meta key.
   */
  public MaxMetaLayer() {
  }

  /**
   * Instantiates a new Max meta key.
   *
   * @param id the id
   */
  protected MaxMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * From json max meta key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the max meta key
   */
  public static MaxMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxMetaLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result input = inObj[0];
    input.addRef();
    final int itemCnt = input.getData().length();
    final Tensor input0Tensor = input.getData().get(0);
    final int vectorSize = input0Tensor.length();
    @Nonnull final int[] indicies = new int[vectorSize];
    for (int i = 0; i < vectorSize; i++) {
      final int itemNumber = i;
      indicies[i] = IntStream.range(0, itemCnt)
          .mapToObj(x -> x).max(Comparator.comparing(dataIndex -> {
            Tensor tensor = input.getData().get(dataIndex);
            double v = tensor.getData()[itemNumber];
            tensor.freeRef();
            return v;
          })).get();
    }
    return new Result(TensorArray.wrap(input0Tensor.mapIndex((v, c) -> {
      Tensor tensor = input.getData().get(indicies[c]);
      double v1 = tensor.getData()[c];
      tensor.freeRef();
      return v1;
    })), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      if (input.isAlive()) {
        @Nullable final Tensor delta = data.get(0);
        @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
        Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
        input0Tensor.coordStream(true).forEach((inputCoord) -> {
          feedback[indicies[inputCoord.getIndex()]].add(inputCoord, delta.get(inputCoord));
        });
        @Nonnull TensorArray tensorArray = TensorArray.wrap(feedback);
        input.accumulate(buffer, tensorArray);
        delta.freeRef();
      }
    }) {

      @Override
      public boolean isAlive() {
        return input.isAlive();
      }

      @Override
      protected void _free() {
        input.freeRef();
        input0Tensor.freeRef();
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
