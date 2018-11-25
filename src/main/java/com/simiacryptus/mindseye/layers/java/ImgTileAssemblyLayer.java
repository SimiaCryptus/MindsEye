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
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Reduces the resolution of the input by selecting a centered window. The output png will have the same number of
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
   * Instantiates a new Img crop key.
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
   * Instantiates a new Img crop key.
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
   * @param offsetX    the position x
   * @param offsetY    the position y
   * @param paddingX   the padding x
   * @param paddingY   the padding y
   * @param toroidal   the toroidal
   * @param rowF       the row f
   * @param colF       the col f
   * @return the tensor
   */
  @Nonnull
  public static Tensor copy(
      @Nonnull final Tensor inputData,
      @Nonnull final Tensor outputData,
      final int offsetX,
      final int offsetY,
      final int paddingX,
      final int paddingY,
      final boolean toroidal,
      final double rowF,
      final double colF
  ) {
    int[] inputDataDimensions = inputData.getDimensions();
    @Nonnull final int[] inDim = inputDataDimensions;
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : Arrays.toString(inDim) + "; " + Arrays.toString(outDim);
//    outputData.coordStream(true).forEach((outputCoord) -> {
//      double value = getValue(inputData, outputCoord, offsetX, offsetY, paddingX, paddingY, toroidal);
//      if (Double.isFinite(value)) outputData.set(outputCoord, value);
//    });
    inputData.coordStream(true).forEach(inputCoord -> {
      double inputValue = inputData.get(inputCoord);
      int[] outputDataDimensions = outputData.getDimensions();
      int inputWidth = inputDataDimensions[0];
      int inputHeight = inputDataDimensions[1];
      int outputWidth = outputDataDimensions[0];
      int outputHeight = outputDataDimensions[1];
      int x = inputCoord.getCoords()[0];
      int y = inputCoord.getCoords()[1];
//      x += offsetX;
//      y += offsetY;
      if (x < paddingX / 2 && colF > 0.0) {
        return;
      }
      if (y < paddingY / 2 && rowF > 0.0) {
        return;
      }
      if (x >= inputWidth - paddingX / 2 && colF < 1.0) {
        return;
      }
      if (y >= inputHeight - paddingY / 2 && rowF < 1.0) {
        return;
      }
      x += offsetX;
      y += offsetY;
      int z = inputCoord.getCoords()[2];
      if (toroidal) {
        while (x < 0) x += outputWidth;
        x %= outputWidth;
        while (y < 0) y += outputHeight;
        y %= outputHeight;
      }
      if (x < 0) {
        return;
      }
      if (y < 0) {
        return;
      }
      if (x >= outputWidth) {
        return;
      }
      if (y >= outputHeight) {
        return;
      }
      outputData.set(x, y, z, inputValue);
    });

    return outputData;
  }


  /**
   * From json img crop key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img crop key
   */
  public static ImgTileAssemblyLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
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
          int positionY = offsetY;
          int inputIndex = 0;
          for (int row = 0; row < rows; row++) {
            int positionX = offsetX;
            int rowHeight = 0;
            for (int col = 0; col < columns; col++) {
              TensorList tileTensor = inObj[inputIndex].getData();
              int[] tileDimensions = tileTensor.getDimensions();
              rowHeight = Math.max(rowHeight, tileDimensions[1]);
              Tensor inputData = tileTensor.get(dataIndex);
              ImgTileAssemblyLayer.copy(
                  inputData,
                  outputData,
                  positionX,
                  positionY,
                  0 >= positionX ? 0 : getPaddingX() / 2,
                  0 >= positionY ? 0 : getPaddingY() / 2,
                  offsetX < 0 || offsetY < 0,
                  (double) row / (rows - 1),
                  (double) col / (columns - 1)
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
        .toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      final AtomicInteger positionY = new AtomicInteger(offsetX);
      int inputIndex = 0;
      for (int row = 0; row < rows; row++) {
        final AtomicInteger positionX = new AtomicInteger(offsetY);
        int rowHeight = 0;
        for (int col = 0; col < columns; col++) {
          Result in = inObj[inputIndex++];
          int[] inputDataDimensions = in.getData().getDimensions();
          rowHeight = Math.max(rowHeight, inputDataDimensions[1]);
          if (in.isAlive()) {
            final int finalRow = row;
            final int finalCol = col;
            @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).parallel()
                .mapToObj(dataIndex -> {
                  @Nullable final Tensor deltaTensor = delta.get(dataIndex);
                  @Nonnull final Tensor passbackTensor = new Tensor(inputDataDimensions);
                  ImgTileAssemblyLayer.copy(deltaTensor, passbackTensor,
                      -positionX.get(),
                      -positionY.get(),
                      0 == positionX.get() ? 0 : getPaddingX() / 2,
                      0 == positionY.get() ? 0 : getPaddingY() / 2,
                      offsetX < 0 || offsetY < 0,
                      (double) finalRow / rows,
                      (double) finalCol / columns
                  );
                  deltaTensor.freeRef();
                  return passbackTensor;
                }).toArray(i -> new Tensor[i]));
            in.accumulate(buffer, tensorArray);
          }
          positionX.addAndGet(inputDataDimensions[0] - getPaddingX());
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
        //positionX += dimensions[0] - positionX==0?0:getPaddingX();
        positionX += dimensions[0] - getPaddingX();
        inputIndex += 1;
      }
//      totalHeight += rowHeight - totalHeight==0?0:getPaddingY();
      totalHeight += rowHeight - getPaddingY();
      totalWidth = Math.max(totalWidth, positionX);
    }
    return new int[]{totalWidth + getPaddingX(), totalHeight + getPaddingY(), bands};
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


  /**
   * Gets padding x.
   *
   * @return the padding x
   */
  public int getPaddingX() {
    return paddingX;
  }

  /**
   * Sets padding x.
   *
   * @param paddingX the padding x
   * @return the padding x
   */
  public ImgTileAssemblyLayer setPaddingX(int paddingX) {
    this.paddingX = paddingX;
    return this;
  }

  /**
   * Gets padding y.
   *
   * @return the padding y
   */
  public int getPaddingY() {
    return paddingY;
  }

  /**
   * Sets padding y.
   *
   * @param paddingY the padding y
   * @return the padding y
   */
  public ImgTileAssemblyLayer setPaddingY(int paddingY) {
    this.paddingY = paddingY;
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
   * @return the offset x
   */
  public ImgTileAssemblyLayer setOffsetX(int offsetX) {
    this.offsetX = offsetX;
    return this;
  }

  /**
   * Gets offset y.
   *
   * @return the offset y
   */
  public int getOffsetY() {
    return offsetY;
  }

  /**
   * Sets offset y.
   *
   * @param offsetY the offset y
   * @return the offset y
   */
  public ImgTileAssemblyLayer setOffsetY(int offsetY) {
    this.offsetY = offsetY;
    return this;
  }
}
