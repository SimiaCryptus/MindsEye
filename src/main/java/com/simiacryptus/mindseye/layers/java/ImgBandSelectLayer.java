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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Selects specific color bands from the input, producing an png apply the same resolution but fewer bands.
 */
@SuppressWarnings("serial")
public class ImgBandSelectLayer extends LayerBase {


  private final int[] bands;

  /**
   * Instantiates a new Img band select layer.
   *
   * @param bands the bands
   */
  public ImgBandSelectLayer(final int... bands) {
    super();
    this.bands = bands;
  }

  /**
   * Instantiates a new Img band select layer.
   *
   * @param json the json
   */
  protected ImgBandSelectLayer(@Nonnull final JsonObject json) {
    super(json);
    final JsonArray jsonArray = json.getAsJsonArray("bands");
    bands = new int[jsonArray.size()];
    for (int i = 0; i < bands.length; i++) {
      bands[i] = jsonArray.get(i).getAsInt();
    }
  }

  /**
   * From json img band select layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img band select layer
   */
  public static ImgBandSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandSelectLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    @Nonnull final Tensor outputDims = new Tensor(inputDims[0], inputDims[1], bands.length);
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    @Nonnull TensorArray wrap = TensorArray.wrap(IntStream.range(0, batch.length()).parallel()
        .mapToObj(dataIndex -> outputDims.mapCoords((c) -> {
          int[] coords = c.getCoords();
          @Nullable Tensor tensor = batch.get(dataIndex);
          double v = tensor.get(coords[0], coords[1], bands[coords[2]]);
          tensor.freeRef();
          return v;
        }))
        .toArray(i -> new Tensor[i]));
    outputDims.freeRef();
    return new Result(wrap, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList error) -> {
      if (input.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, error.length()).parallel()
            .mapToObj(dataIndex -> {
              @Nonnull final Tensor passback = new Tensor(inputDims);
              @Nullable final Tensor err = error.get(dataIndex);
              err.coordStream(false).forEach(c -> {
                int[] coords = c.getCoords();
                passback.set(coords[0], coords[1], bands[coords[2]], err.get(c));
              });
              err.freeRef();
              return passback;
            }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }

      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    @Nonnull final JsonArray array = new JsonArray();
    for (final int b : bands) {
      array.add(new JsonPrimitive(b));
    }
    json.add("bands", array);
    return json;
  }


  @Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }


}
