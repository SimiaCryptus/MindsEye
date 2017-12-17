/*
 * Copyright (c) 2017 by Andrew Charneski.
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
import com.simiacryptus.mindseye.lang.NNExecutionContext;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Assert dimensions layer.
 */
@SuppressWarnings("serial")
public class AssertDimensionsLayer extends NNLayer {
  
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
  protected AssertDimensionsLayer(final JsonObject json) {
    super(json);
    final JsonArray dimsJson = json.get("dims").getAsJsonArray();
    dims = IntStream.range(0, dimsJson.size()).map(i -> dimsJson.get(i).getAsInt()).toArray();
  }
  
  /**
   * From json assert dimensions layer.
   *
   * @param json the json
   * @return the assert dimensions layer
   */
  public static AssertDimensionsLayer fromJson(final JsonObject json) {
    return new AssertDimensionsLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... array) {
    if (0 == array.length) {
      throw new IllegalArgumentException();
    }
    if (0 == array[0].getData().length()) {
      throw new IllegalArgumentException();
    }
    final int[] inputDims = array[0].getData().get(0).getDimensions();
    if (!Arrays.equals(inputDims, dims)) {
      throw new IllegalArgumentException(Arrays.toString(inputDims) + " != " + Arrays.toString(dims));
    }
    return array[0];
  }
  
  @Override
  public List<NNLayer> getChildren() {
    return super.getChildren();
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    final JsonArray dimsJson = new JsonArray();
    for (final int dim : dims) {
      dimsJson.add(new JsonPrimitive(dim));
    }
    json.add("dims", dimsJson);
    return json;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
