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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.TensorList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This key works as a scaling function, similar to a father wavelet. Allows convolutional and pooling layers to work
 * across larger png regions.
 */
@SuppressWarnings("serial")
public class ImgTileSubnetLayer extends WrapperLayer {

  private final int height;
  private final int width;
  private final int strideX;
  private final int strideY;

  /**
   * Instantiates a new Rescaled subnet key.
   *
   * @param subnetwork the subnetwork
   * @param width      the width
   * @param height     the scale
   * @param strideX    the stride x
   * @param strideY    the stride y
   */
  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height, final int strideX, final int strideY) {
    super(subnetwork);
    this.height = height;
    this.width = width;
    this.strideX = strideX;
    this.strideY = strideY;
  }

  /**
   * Instantiates a new Img tile subnet key.
   *
   * @param subnetwork the subnetwork
   * @param width      the width
   * @param height     the height
   */
  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height) {
    this(subnetwork, width, height, width, height);
  }

  /**
   * Instantiates a new Rescaled subnet key.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgTileSubnetLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    height = json.getAsJsonPrimitive("height").getAsInt();
    width = json.getAsJsonPrimitive("width").getAsInt();
    strideX = json.getAsJsonPrimitive("strideX").getAsInt();
    strideY = json.getAsJsonPrimitive("strideY").getAsInt();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
  }

  /**
   * From json rescaled subnet key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the rescaled subnet key
   */
  public static ImgTileSubnetLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSubnetLayer(json, rs);
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Result input = inObj[0];
    final TensorList inputData = input.getData();
    @Nonnull final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    int cols = (int) (Math.ceil((inputDims[0] - width) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height) * 1.0 / strideY) + 1);
    if (cols == 1 && rows == 1) return getInner().evalAndFree(inObj);
    Result[] results = new Result[rows * cols];
    TensorList[] passback = new TensorList[rows * cols];
    int index = 0;
    AtomicInteger passbacks = new AtomicInteger(0);
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        input.addRef();
        inputData.addRef();
        int positionX = col * strideX;
        int positionY = row * strideY;
        assert positionX >= 0;
        assert positionY >= 0;
        assert positionX < inputDims[0];
        assert positionY < inputDims[1];
        final int finalIndex = index;
        ImgTileSelectLayer tileSelectLayer = new ImgTileSelectLayer(width, height, positionX, positionY);
        Result selectedTile = tileSelectLayer.evalAndFree(new Result(inputData, (ctx, delta) -> {
          passback[finalIndex] = delta;
          if (passbacks.incrementAndGet() == rows * cols) {
            passbacks.set(0);
            ImgTileAssemblyLayer imgTileAssemblyLayer = new ImgTileAssemblyLayer(cols, rows);
            TensorList reassembled = imgTileAssemblyLayer.evalAndFree(Arrays.stream(passback).map(t -> new Result(t, (c2, d2) -> {
            })).toArray(i -> new Result[i])).getDataAndFree();
            imgTileAssemblyLayer.freeRef();
            input.accumulate(ctx, reassembled);
          }
        }) {
          @Override
          protected void _free() {
            input.freeRef();
            super._free();
          }
        });
        tileSelectLayer.freeRef();
        results[index] = getInner().evalAndFree(selectedTile);
        index = index + 1;
      }
    }
    input.freeRef();
    inputData.freeRef();
    ImgTileAssemblyLayer imgTileAssemblyLayer = new ImgTileAssemblyLayer(cols, rows);
    Result result = imgTileAssemblyLayer.evalAndFree(results);
    imgTileAssemblyLayer.freeRef();
    return result;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("height", height);
    json.addProperty("width", width);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    return json;
  }


  @Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }

}
