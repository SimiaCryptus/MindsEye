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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Increases the resolution of the input by selecting a larger centered window. The output image will have the same
 * number of color bands, and the area outside the source image will be setWeights to 0.
 */
@SuppressWarnings("serial")
public class ImgZeroPaddingLayer extends NNLayer implements MultiPrecision<ImgZeroPaddingLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgZeroPaddingLayer.class);
  
  private int sizeX;
  private int sizeY;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   */
  private ImgZeroPaddingLayer() {
  }
  
  /**
   * Instantiates a new Img zero padding layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgZeroPaddingLayer(int sizeX, int sizeY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgZeroPaddingLayer(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static ImgZeroPaddingLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgZeroPaddingLayer(json, rs);
  }
  
  @Nullable
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    if (sizeX == 0 && sizeY == 0) {
      inObj[0].getData().addRef();
      inObj[0].addRef();
      return inObj[0];
    }
    assert inObj.length == 1;
    @Nonnull int[] dimensions = inObj[0].getData().getDimensions();
    @Nonnull ImgCropLayer imgCropLayer = new ImgCropLayer(dimensions[0] + 2 * this.sizeX, dimensions[1] + 2 * this.sizeY).setPrecision(precision);
    @Nullable NNResult eval = imgCropLayer.eval(inObj);
    imgCropLayer.freeRef();
    return eval;
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeY", sizeY);
    json.addProperty("sizeX", sizeX);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public ImgZeroPaddingLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
}
