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
public class ImgTileAssemblyLayer extends LayerBase {
  
  
  private final int columns;
  private final int rows;
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param columns the size x
   * @param rows    the size y
   */
  public ImgTileAssemblyLayer(final int columns, final int rows) {
    super();
    this.columns = columns;
    this.rows = rows;
  }
  
  /**
   * Instantiates a new Img crop layer.
   *
   * @param json the json
   */
  protected ImgTileAssemblyLayer(@Nonnull final JsonObject json) {
    super(json);
    columns = json.getAsJsonPrimitive("columns").getAsInt();
    rows = json.getAsJsonPrimitive("rows").getAsInt();
  }
  
  /**
   * Copy condense tensor.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @param positionX
   * @param positionY
   * @return the tensor
   */
  @Nonnull
  public static Tensor copy(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData, final int positionX, final int positionY) {
    int[] inputDataDimensions = inputData.getDimensions();
    @Nonnull final int[] inDim = inputDataDimensions;
    @Nonnull final int[] outDim = outputData.getDimensions();
    int width = inputDataDimensions[0];
    int height = inputDataDimensions[1];
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : Arrays.toString(inDim) + "; " + Arrays.toString(outDim);
    outputData.coordStream(true).forEach((c) -> {
      int x = c.getCoords()[0] - positionX;
      int y = c.getCoords()[1] - positionY;
      int z = c.getCoords()[2];
      double value;
      if (x < 0) { value = Double.NaN; }
      else if (x >= width) { value = Double.NaN; }
      else if (y < 0) { value = Double.NaN; }
      else if (y >= height) { value = Double.NaN; }
      else { value = inputData.get(x, y, z); }
      if (Double.isFinite(value)) outputData.set(c, value);
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
  public static ImgTileAssemblyLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgTileAssemblyLayer(json);
  }
  
  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    assert 3 == inObj[0].getData().getDimensions().length;
    int[] outputDims = getOutputDims(inObj);
    return new Result(TensorArray.wrap(IntStream.range(0, inObj[0].getData().length()).parallel()
      .mapToObj(dataIndex -> {
        @Nonnull final Tensor outputData = new Tensor(outputDims);
        
        int totalWidth = 0;
        int totalHeight = 0;
        int inputIndex = 0;
        for (int row = 0; row < rows; row++) {
          int positionX = 0;
          int rowHeight = 0;
          for (int col = 0; col < columns; col++) {
            TensorList tileTensor = inObj[inputIndex].getData();
            int[] tileDimensions = tileTensor.getDimensions();
            rowHeight = Math.max(rowHeight, tileDimensions[1]);
            Tensor inputData = tileTensor.get(dataIndex);
            ImgTileAssemblyLayer.copy(inputData, outputData, positionX, totalHeight);
            inputData.freeRef();
            positionX += tileDimensions[0];
            inputIndex += 1;
          }
          totalHeight += rowHeight;
          totalWidth = Math.max(totalWidth, positionX);
        }
        
        return outputData;
      })
      .toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      int totalHeight = 0;
      int inputIndex = 0;
      for (int row = 0; row < rows; row++) {
        int positionX = 0;
        int rowHeight = 0;
        for (int col = 0; col < columns; col++) {
          Result in = inObj[inputIndex];
          int[] inputDataDimensions = in.getData().getDimensions();
          rowHeight = Math.max(rowHeight, inputDataDimensions[1]);
          if (in.isAlive()) {
            int _positionX = positionX;
            int _totalHeight = totalHeight;
            @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).parallel()
              .mapToObj(dataIndex -> {
                @Nullable final Tensor deltaTensor = delta.get(dataIndex);
                @Nonnull final Tensor passbackTensor = new Tensor(inputDataDimensions);
                ImgTileAssemblyLayer.copy(deltaTensor, passbackTensor, -_positionX, -_totalHeight);
                deltaTensor.freeRef();
                return passbackTensor;
              }).toArray(i -> new Tensor[i]));
            in.accumulate(buffer, tensorArray);
            tensorArray.freeRef();
          }
          positionX += inputDataDimensions[0];
          inputIndex += 1;
        }
        totalHeight += rowHeight;
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    };
  }
  
  private int[] getOutputDims(@Nonnull final Result[] inObj) {
    int bands = inObj[0].getData().getDimensions()[2];
    int totalWidth = 0;
    int totalHeight = 0;
    int inputIndex = 0;
    for (int row = 0; row < rows; row++) {
      int positionX = 0;
      int rowHeight = 0;
      for (int col = 0; col < columns; col++) {
        int[] dimensions = inObj[inputIndex].getData().getDimensions();
        rowHeight = Math.max(rowHeight, dimensions[1]);
        positionX += dimensions[0];
        inputIndex += 1;
      }
      totalHeight += rowHeight;
      totalWidth = Math.max(totalWidth, positionX);
    }
    return new int[]{totalWidth, totalHeight, bands};
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("columns", columns);
    json.addProperty("rows", rows);
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
