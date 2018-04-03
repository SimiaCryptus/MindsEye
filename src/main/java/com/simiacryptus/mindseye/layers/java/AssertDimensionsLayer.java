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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * This layer is a pass-thru except that it throws an error if the dimensions are not consistent apply its setting.
 */
@SuppressWarnings("serial")
public class AssertDimensionsLayer extends LayerBase {
  
  private final int[] dims;
  
  /**
   * Instantiates a new Assert dimensions layer.
   *
   * @param dims the dims
   */
  public AssertDimensionsLayer(final int... dims) {
    super();
    this.dims = dims;
  }
  
  /**
   * Instantiates a new Assert dimensions layer.
   *
   * @param json the json
   */
  protected AssertDimensionsLayer(@Nonnull final JsonObject json) {
    super(json);
    final JsonArray dimsJson = json.get("dims").getAsJsonArray();
    dims = IntStream.range(0, dimsJson.size()).map(i -> dimsJson.get(i).getAsInt()).toArray();
  }
  
  /**
   * From json assert dimensions layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the assert dimensions layer
   */
  public static AssertDimensionsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AssertDimensionsLayer(json);
  }
  
  @Override
  public Result evalAndFree(@Nonnull final Result... array) {
    if (0 == array.length) {
      throw new IllegalArgumentException();
    }
    Result input = array[0];
    if (0 == input.getData().length()) {
      throw new IllegalArgumentException();
    }
    @Nonnull final int[] inputDims = input.getData().getDimensions();
    if (!Arrays.equals(inputDims, dims)) {
      throw new IllegalArgumentException(Arrays.toString(inputDims) + " != " + Arrays.toString(dims));
    }
    return input;
  }
  
  @Override
  public List<Layer> getChildren() {
    return super.getChildren();
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    @Nonnull final JsonArray dimsJson = new JsonArray();
    for (final int dim : dims) {
      dimsJson.add(new JsonPrimitive(dim));
    }
    json.add("dims", dimsJson);
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
