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
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Reduces the resolution of the input by selecting a centered window. The output png will have the same number of
 * color bands.
 */
@SuppressWarnings("serial")
public class ImgTileSelectLayer extends LayerBase {


  private final boolean toroidal;
  private final int sizeX;
  private final int sizeY;
  private final int positionX;
  private final int positionY;

  /**
   * Instantiates a new Img crop key.
   *
   * @param sizeX     the size x
   * @param sizeY     the size y
   * @param positionX the position x
   * @param positionY the position y
   */
  public ImgTileSelectLayer(final int sizeX, final int sizeY, final int positionX, final int positionY) {
    this(
        sizeX,
        sizeY,
        positionX,
        positionY,
        false
    );
  }

  /**
   * Instantiates a new Img crop key.
   *
   * @param sizeX     the size x
   * @param sizeY     the size y
   * @param positionX the position x
   * @param positionY the position y
   * @param toroidal  the toroidal
   */
  public ImgTileSelectLayer(final int sizeX, final int sizeY, final int positionX, final int positionY, final boolean toroidal) {
    super();
    this.sizeX = sizeX;
    this.sizeY = sizeY;
    this.positionX = positionX;
    this.positionY = positionY;
    this.toroidal = toroidal;
  }

  /**
   * Instantiates a new Img crop key.
   *
   * @param json the json
   */
  protected ImgTileSelectLayer(@Nonnull final JsonObject json) {
    super(json);
    sizeX = json.getAsJsonPrimitive("sizeX").getAsInt();
    sizeY = json.getAsJsonPrimitive("sizeY").getAsInt();
    positionX = json.getAsJsonPrimitive("positionX").getAsInt();
    positionY = json.getAsJsonPrimitive("positionY").getAsInt();
    toroidal = json.getAsJsonPrimitive("toroidal").getAsBoolean();
  }

  /**
   * Copy condense tensor.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @param posX       the pos x
   * @param posY       the pos y
   * @param toroidal   the toroidal
   * @return the tensor
   */
  @Nonnull
  public static Tensor copy(
      @Nonnull final Tensor inputData,
      @Nonnull final Tensor outputData,
      final int posX,
      final int posY,
      final boolean toroidal
  ) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : Arrays.toString(inDim) + "; " + Arrays.toString(outDim);
    outputData.coordStream(false).forEach((c) -> {
      int x = c.getCoords()[0] + posX;
      int y = c.getCoords()[1] + posY;
      int z = c.getCoords()[2];
      int width = inputData.getDimensions()[0];
      int height = inputData.getDimensions()[1];
      if (toroidal) {
        while (x < 0) x += width;
        x %= width;
        while (y < 0) y += height;
        y %= height;
      }
      double value;
      if (x < 0) {
        value = 0.0;
      } else if (x >= width) {
        value = 0.0;
      } else if (y < 0) {
        value = 0.0;
      } else if (y >= height) {
        value = 0.0;
      } else {
        value = inputData.get(x, y, z);
      }
      outputData.set(c, value);
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
  public static ImgTileSelectLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgTileSelectLayer(json);
  }

  /**
   * To tiles tensor [ ].
   *
   * @param log     the log
   * @param canvas  the canvas
   * @param width   the width
   * @param height  the height
   * @param strideX the stride x
   * @param strideY the stride y
   * @param offsetX the offset x
   * @param offsetY the offset y
   * @return the tensor [ ]
   */
  @Nonnull
  public static Tensor[] toTiles(
      final NotebookOutput log,
      final Tensor canvas,
      final int width,
      final int height,
      final int strideX,
      final int strideY,
      final int offsetX,
      final int offsetY
  ) {

    @Nonnull final int[] inputDims = canvas.getDimensions();
    int cols = (int) (Math.ceil((inputDims[0] - width - offsetX) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height - offsetY) * 1.0 / strideY) + 1);
    log.p(String.format(
        "Partition %s x %s png with %s x %s tile size into %s x %s grid with stride %s x %s offset %s x %s",
        inputDims[0],
        inputDims[1],
        width,
        height,
        cols,
        rows,
        strideX,
        strideY,
        offsetX,
        offsetY
    ));
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
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    @Nonnull final int[] dimOut = getViewDimensions(inputDims, new int[]{sizeX, sizeY, inputDims[2]}, new int[]{positionX, positionY, 0});
    return new Result(TensorArray.wrap(IntStream.range(0, batch.length()).parallel()
        .mapToObj(dataIndex -> {
          @Nonnull final Tensor outputData = new Tensor(dimOut);
          Tensor inputData = batch.get(dataIndex);
          copy(inputData, outputData, positionX, positionY, toroidal);
          inputData.freeRef();
          return outputData;
        })
        .toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList error) -> {
      if (input.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, error.length()).parallel()
            .mapToObj(dataIndex -> {
              @Nullable final Tensor err = error.get(dataIndex);
              @Nonnull final Tensor passback = new Tensor(inputDims);
              copy(err, passback, -positionX, -positionY, toroidal);
              err.freeRef();
              return passback;
            }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
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

  /**
   * Get view dimensions int [ ].
   *
   * @param sourceDimensions      the source dimensions
   * @param destinationDimensions the destination dimensions
   * @param offset                the offset
   * @return the int [ ]
   */
  @Nonnull
  public int[] getViewDimensions(int[] sourceDimensions, int[] destinationDimensions, int[] offset) {
    @Nonnull final int[] viewDim = new int[3];
    Arrays.parallelSetAll(viewDim, i -> toroidal ? (destinationDimensions[i]) : (Math.min(
        sourceDimensions[i],
        destinationDimensions[i] + offset[i]
    ) - Math.max(offset[i], 0)));
    return viewDim;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeY);
    json.addProperty("positionX", positionX);
    json.addProperty("positionY", positionY);
    json.addProperty("toroidal", toroidal);
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }


}
