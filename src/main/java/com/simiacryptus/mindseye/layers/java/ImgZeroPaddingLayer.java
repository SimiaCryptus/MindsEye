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
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Increases the resolution of the input by selecting a larger centered window. The output image will have the same
 * number of color bands, and the area outside the source image will be setWeights to 0.
 */
@SuppressWarnings("serial")
public class ImgZeroPaddingLayer extends NNLayer {
  
  
  private final int sizeX;
  private final int sizeY;
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgZeroPaddingLayer(final int sizeX, final int sizeY) {
    super();
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param json the json
   */
  protected ImgZeroPaddingLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    sizeX = json.getAsJsonPrimitive("sizeX").getAsInt();
    sizeY = json.getAsJsonPrimitive("sizeY").getAsInt();
  }
  
  /**
   * From json img crop layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img crop layer
   */
  public static ImgZeroPaddingLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgZeroPaddingLayer(json);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    assert inObj.length == 1;
    int[] dimensions = inObj[0].getData().getDimensions();
    return new ImgCropLayer(dimensions[0] + 2 * this.sizeX, dimensions[1] + 2 * this.sizeY).eval(inObj);
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeX);
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
