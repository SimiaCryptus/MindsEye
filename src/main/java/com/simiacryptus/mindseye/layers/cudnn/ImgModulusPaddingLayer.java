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
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
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
public class ImgModulusPaddingLayer extends LayerBase implements MultiPrecision<ImgModulusPaddingLayer> {
  private static final Logger log = LoggerFactory.getLogger(ImgModulusPaddingLayer.class);
  
  private int sizeX;
  private int sizeY;
  private int offsetX;
  private int offsetY;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   */
  private ImgModulusPaddingLayer() {
  }
  
  /**
   * Instantiates a new Img zero padding layer.
   *
   * @param sizeX   the size x
   * @param sizeY   the size y
   * @param offsetX the offset x
   * @param offsetY the offset y
   */
  public ImgModulusPaddingLayer(int sizeX, int sizeY, int offsetX, int offsetY) {
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    this.offsetX = offsetX;
    this.offsetY = offsetY;
  }
  
  /**
   * Instantiates a new Img modulus padding layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgModulusPaddingLayer(int sizeX, int sizeY) {
    this(sizeX, sizeY, 0, 0);
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgModulusPaddingLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    sizeX = json.get("sizeX").getAsInt();
    sizeY = json.get("sizeY").getAsInt();
    offsetX = json.get("offsetX").getAsInt();
    offsetY = json.get("offsetY").getAsInt();
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static ImgModulusPaddingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgModulusPaddingLayer(json, rs);
  }
  
  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    assert inObj.length == 1;
    @Nonnull int[] dimensions = inObj[0].getData().getDimensions();
    int inputWidth = dimensions[0];
    int inputHeight = dimensions[1];
  
    int sizeX = Math.abs(this.sizeX);
    int paddingX = sizeX - ((inputWidth - offsetX) % sizeX);
    while (paddingX < 0) paddingX += sizeX;
    while (paddingX >= sizeX) paddingX -= sizeX;
    if (this.sizeX < 0 && (paddingX + inputWidth) > sizeX) paddingX -= sizeX;
  
    int sizeY = Math.abs(this.sizeY);
    int paddingY = sizeY - ((inputHeight - offsetY) % sizeY);
    while (paddingY < 0) paddingY += sizeY;
    while (paddingY >= sizeY) paddingY -= sizeY;
    if (this.sizeY < 0 && (paddingY + inputHeight) > sizeY) paddingY -= sizeY;
    
    int ouputWidth = inputWidth + paddingX;
    int outputHeight = inputHeight + paddingY;
    assert ouputWidth > 0;
    assert outputHeight > 0;
    if (ouputWidth == inputWidth) {
      if (outputHeight == inputHeight) {
        return inObj[0];
      }
    }
    
    @Nonnull ImgCropLayer imgCropLayer = new ImgCropLayer(ouputWidth, outputHeight).setPrecision(precision);
    @Nullable Result eval = imgCropLayer.evalAndFree(inObj);
    imgCropLayer.freeRef();
    return eval;
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeY", sizeY);
    json.addProperty("sizeX", sizeX);
    json.addProperty("offsetX", offsetX);
    json.addProperty("offsetY", offsetY);
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @Nonnull
  @Override
  public ImgModulusPaddingLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
  
  /**
   * Gets offset x.
   *
   * @return the offset x
   */
  public int getOffsetX() {
    return offsetX;
  }
  
  /**
   * Sets offset x.
   *
   * @param offsetX the offset x
   */
  public void setOffsetX(int offsetX) {
    this.offsetX = offsetX;
  }
}
