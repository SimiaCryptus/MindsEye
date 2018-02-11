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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Reduces the resolution of the input by selecting a centered window. The output image will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgCropLayer extends NNLayer {
  
  
  private final int sizeX;
  private final int sizeY;
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgCropLayer(final int sizeX, final int sizeY) {
    super();
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param json the json
   */
  protected ImgCropLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    sizeX = json.getAsJsonPrimitive("sizeX").getAsInt();
    sizeY = json.getAsJsonPrimitive("sizeY").getAsInt();
  }
  
  /**
   * Copy condense tensor.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @return the tensor
   */
  @javax.annotation.Nonnull
  public static Tensor copy(@javax.annotation.Nonnull final Tensor inputData, @javax.annotation.Nonnull final Tensor outputData) {
    @javax.annotation.Nonnull final int[] inDim = inputData.getDimensions();
    @javax.annotation.Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : Arrays.toString(inDim) + "; " + Arrays.toString(outDim);
    double fx = (inDim[0] - outDim[0]) / 2.0;
    double fy = (inDim[1] - outDim[1]) / 2.0;
    final int paddingX = (int) (fx < 0 ? Math.ceil(fx) : Math.floor(fx));
    final int paddingY = (int) (fy < 0 ? Math.ceil(fy) : Math.floor(fy));
    outputData.coordStream(true).forEach((c) -> {
      int x = c.getCoords()[0] + paddingX;
      int y = c.getCoords()[1] + paddingY;
      int z = c.getCoords()[2];
      int width = inputData.getDimensions()[0];
      int height = inputData.getDimensions()[1];
      double value;
      if (x < 0) { value = 0.0; }
      else if (x >= width) { value = 0.0; }
      else if (y < 0) { value = 0.0; }
      else if (y >= height) { value = 0.0; }
      else { value = inputData.get(x, y, z); }
      outputData.set(c, value);
    });
    return outputData;
  }
  
  /**
   * From json img crop layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img crop layer
   */
  public static ImgCropLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgCropLayer(json);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    return new NNResult(TensorArray.wrap(IntStream.range(0, batch.length()).parallel()
      .mapToObj(dataIndex -> {
        @javax.annotation.Nonnull final Tensor outputData = new Tensor(sizeX, sizeY, inputDims[2]);
        Tensor inputData = batch.get(dataIndex);
        ImgCropLayer.copy(inputData, outputData);
        inputData.freeRef();
        return outputData;
      })
      .toArray(i -> new Tensor[i])), (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList error) -> {
      if (input.isAlive()) {
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, error.length()).parallel()
          .mapToObj(dataIndex -> {
            @Nullable final Tensor err = error.get(dataIndex);
            @javax.annotation.Nonnull final Tensor passback = new Tensor(inputDims);
            copy(err, passback);
            err.freeRef();
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
