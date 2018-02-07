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
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Selects specific color bands from the input, producing an image with the same resolution but fewer bands.
 */
@SuppressWarnings("serial")
public class ImgBandSelectLayer extends NNLayer {
  
  
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
  protected ImgBandSelectLayer(final @NotNull JsonObject json) {
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
  public static ImgBandSelectLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new ImgBandSelectLayer(json);
  }
  
  @Override
  public @NotNull NNResult eval(final @NotNull NNResult... inObj) {
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final @NotNull int[] inputDims = batch.get(0).getDimensions();
    assert 3 == inputDims.length;
    final @NotNull Tensor outputDims = new Tensor(inputDims[0], inputDims[1], bands.length);
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    return new NNResult(TensorArray.wrap(IntStream.range(0, batch.length()).parallel()
                                                  .mapToObj(dataIndex -> outputDims.mapCoords((c) -> {
                                                    int[] coords = c.getCoords();
                                                    return batch.get(dataIndex).get(coords[0], coords[1], bands[coords[2]]);
                                                  }))
                                                  .toArray(i -> new Tensor[i])), (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList error) -> {
      if (input.isAlive()) {
        @NotNull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, error.length()).parallel()
                                                                     .mapToObj(dataIndex -> {
                                                                       final @NotNull Tensor passback = new Tensor(inputDims);
                                                              final Tensor err = error.get(dataIndex);
                                                              err.coordStream(false).forEach(c -> {
                                                                int[] coords = c.getCoords();
                                                                passback.set(coords[0], coords[1], bands[coords[2]], err.get(c));
                                                              });
                                                              return passback;
                                                            }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
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
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final @NotNull JsonObject json = super.getJsonStub();
    final @NotNull JsonArray array = new JsonArray();
    for (final int b : bands) {
      array.add(new JsonPrimitive(b));
    }
    json.add("bands", array);
    return json;
  }
  
  
  @Override
  public @NotNull List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
