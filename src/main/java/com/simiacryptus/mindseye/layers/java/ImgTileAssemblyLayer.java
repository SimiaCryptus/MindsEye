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
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Reduces the resolution of the input by selecting a centered window. The output image will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgTileAssemblyLayer extends LayerBase {
  
  
  private final int columns;
  private final int rows;
  private int paddingX = 0;
  private int paddingY = 0;
  private int offsetX = 0;
  private int offsetY = 0;
  
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
    setPaddingX(json.getAsJsonPrimitive("paddingX").getAsInt());
    setPaddingY(json.getAsJsonPrimitive("paddingY").getAsInt());
    setOffsetX(json.getAsJsonPrimitive("offsetX").getAsInt());
    setOffsetY(json.getAsJsonPrimitive("offsetY").getAsInt());
  }
  
  /**
   * Copy condense tensor.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @param outputX  the position x
   * @param outputY  the position y
   * @param inputX
   * @param inputY
   * @param toroidal
   * @return the tensor
   */
  @Nonnull
  public static Tensor copy(
    @Nonnull final Tensor inputData,
    @Nonnull final Tensor outputData,
    final int outputX,
    final int outputY,
    final int inputX,
    final int inputY,
    final boolean toroidal
  )
  {
    int[] inputDataDimensions = inputData.getDimensions();
    @Nonnull final int[] inDim = inputDataDimensions;
    @Nonnull final int[] outDim = outputData.getDimensions();
    int inputWidth = inputDataDimensions[0];
    int inputHeight = inputDataDimensions[1];
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : Arrays.toString(inDim) + "; " + Arrays.toString(outDim);
    outputData.coordStream(true).forEach((c) -> {
      double value = getPixelValue(inputData, outputX, outputY, toroidal, inputWidth, inputHeight, c);
      if (Double.isFinite(value)) outputData.set(c, value);
    });
    return outputData;
  }
  
  public static double getPixelValue(
    @Nonnull final Tensor inputData,
    final int outputX,
    final int outputY,
    final boolean toroidal,
    final int inputWidth,
    final int inputHeight,
    final Coordinate c
  )
  {
    int x = c.getCoords()[0] - outputX;
    int y = c.getCoords()[1] - outputY;
    int z = c.getCoords()[2];
    if (toroidal) {
      while (x < 0) x += inputWidth;
      x %= inputWidth;
      while (y < 0) y += inputHeight;
      y %= inputHeight;
    }
    double value;
    if (x < 0) { value = Double.NaN; }
    else if (x >= inputWidth) { value = Double.NaN; }
    else if (y < 0) { value = Double.NaN; }
    else if (y >= inputHeight) { value = Double.NaN; }
    else { value = inputData.get(x, y, z); }
    return value;
  }
  
  /**
   * From json img crop layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img crop layer
   */
  public static ImgTileAssemblyLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileAssemblyLayer(json);
  }
  
  /**
   * To tiles tensor [ ].
   *
   * @param canvas  the canvas
   * @param width   the width
   * @param height  the height
   * @param strideX the stride x
   * @param strideY the stride y
   * @return the tensor [ ]
   */
  @Nonnull
  public static Tensor[] toTiles(
    final Tensor canvas,
    final int width,
    final int height,
    final int strideX,
    final int strideY,
    final int offsetX,
    final int offsetY
  )
  {
    @Nonnull final int[] inputDims = canvas.getDimensions();
    int cols = (int) (Math.ceil((inputDims[0] - width - offsetX) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height - offsetY) * 1.0 / strideY) + 1);
    Tensor[] tiles = new Tensor[rows * cols];
    int index = 0;
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        int positionX = col * strideX + offsetX;
        int positionY = row * strideY + offsetY;
        ImgTileSelectLayer tileSelectLayer = new ImgTileSelectLayer(width, height, positionX, positionY, offsetX < 0 || offsetY < 0);
        tiles[index++] = tileSelectLayer.eval(canvas).getDataAndFree().getAndFree(0);
        tileSelectLayer.freeRef();
      }
    }
    return tiles;
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
        int positionY = -offsetY;
        int inputIndex = 0;
        for (int row = 0; row < rows; row++) {
          int positionX = -offsetX;
          int rowHeight = 0;
          for (int col = 0; col < columns; col++) {
            TensorList tileTensor = inObj[inputIndex].getData();
            int[] tileDimensions = tileTensor.getDimensions();
            rowHeight = Math.max(rowHeight, tileDimensions[1]);
            Tensor inputData = tileTensor.get(dataIndex);
            ImgTileAssemblyLayer.copy(inputData, outputData,
                                      positionX, positionY,
                                      0 == positionX ? 0 : getPaddingX() / 2, 0 == positionY ? 0 : getPaddingY() / 2,
                                      offsetX < 0 || offsetY < 0
            );
            inputData.freeRef();
            positionX += tileDimensions[0] - getPaddingX();
            inputIndex += 1;
          }
          positionY += rowHeight - getPaddingY();
          totalWidth = Math.max(totalWidth, positionX);
        }
        
        return outputData;
      })
      .toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      final AtomicInteger positionY = new AtomicInteger(offsetX);
      int inputIndex = 0;
      for (int row = 0; row < rows; row++) {
        final AtomicInteger positionX = new AtomicInteger(offsetY);
        int rowHeight = 0;
        for (int col = 0; col < columns; col++) {
          Result in = inObj[inputIndex];
          int[] inputDataDimensions = in.getData().getDimensions();
          rowHeight = Math.max(rowHeight, inputDataDimensions[1]);
          if (in.isAlive()) {
            @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).parallel()
              .mapToObj(dataIndex -> {
                @Nullable final Tensor deltaTensor = delta.get(dataIndex);
                @Nonnull final Tensor passbackTensor = new Tensor(inputDataDimensions);
                ImgTileAssemblyLayer.copy(deltaTensor, passbackTensor,
                                          -positionX.get(),
                                          -positionY.get(),
                                          0 == positionX.get() ? 0 : getPaddingX() / 2,
                                          0 == positionY.get() ? 0 : getPaddingY() / 2,
                                          offsetX < 0 || offsetY < 0
                );
                deltaTensor.freeRef();
                return passbackTensor;
              }).toArray(i -> new Tensor[i]));
            in.accumulate(buffer, tensorArray);
          }
          positionX.addAndGet(inputDataDimensions[0] - getPaddingX());
          inputIndex += 1;
        }
        positionY.addAndGet(rowHeight - getPaddingY());
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
    int[] dimensions1 = inObj[0].getData().getDimensions();
    int bands = dimensions1.length < 2 ? 1 : dimensions1[2];
    int totalWidth = 0;
    int totalHeight = 0;
    int inputIndex = 0;
    for (int row = 0; row < rows; row++) {
      int positionX = 0;
      int rowHeight = 0;
      for (int col = 0; col < columns; col++) {
        int[] dimensions = inObj[inputIndex].getData().getDimensions();
        rowHeight = Math.max(rowHeight, dimensions[1]);
        positionX += dimensions[0] - getPaddingX();
        inputIndex += 1;
      }
      totalHeight += rowHeight - getPaddingY();
      totalWidth = Math.max(totalWidth, positionX);
    }
    return new int[]{totalWidth, totalHeight, bands};
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("columns", columns);
    json.addProperty("rows", rows);
    json.addProperty("paddingX", getPaddingX());
    json.addProperty("paddingY", getPaddingY());
    json.addProperty("offsetX", getOffsetX());
    json.addProperty("offsetY", getOffsetY());
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
  public int getPaddingX() {
    return paddingX;
  }
  
  public ImgTileAssemblyLayer setPaddingX(int paddingX) {
    this.paddingX = paddingX;
    return this;
  }
  
  public int getPaddingY() {
    return paddingY;
  }
  
  public ImgTileAssemblyLayer setPaddingY(int paddingY) {
    this.paddingY = paddingY;
    return this;
  }
  
  public int getOffsetX() {
    return offsetX;
  }
  
  public ImgTileAssemblyLayer setOffsetX(int offsetX) {
    this.offsetX = offsetX;
    return this;
  }
  
  public int getOffsetY() {
    return offsetY;
  }
  
  public ImgTileAssemblyLayer setOffsetY(int offsetY) {
    this.offsetY = offsetY;
    return this;
  }
}
