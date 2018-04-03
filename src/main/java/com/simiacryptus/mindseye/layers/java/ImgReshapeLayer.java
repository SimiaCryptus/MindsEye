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
import java.util.stream.IntStream;

/**
 * Reduces or expands image resolution by rearranging the values in NxN tiles to effectively stripe the small-scale
 * spacial dimension across N^2 color bands.
 */
@SuppressWarnings("serial")
public class ImgReshapeLayer extends LayerBase {
  
  
  private final boolean expand;
  private final int kernelSizeX;
  private final int kernelSizeY;
  
  /**
   * Instantiates a new Img reshapeCast layer.
   *
   * @param kernelSizeX the kernel size x
   * @param kernelSizeY the kernel size y
   * @param expand      the expand
   */
  public ImgReshapeLayer(final int kernelSizeX, final int kernelSizeY, final boolean expand) {
    super();
    this.kernelSizeX = kernelSizeX;
    this.kernelSizeY = kernelSizeY;
    this.expand = expand;
  }
  
  /**
   * Instantiates a new Img reshapeCast layer.
   *
   * @param json the json
   */
  protected ImgReshapeLayer(@Nonnull final JsonObject json) {
    super(json);
    kernelSizeX = json.getAsJsonPrimitive("kernelSizeX").getAsInt();
    kernelSizeY = json.getAsJsonPrimitive("kernelSizeY").getAsInt();
    expand = json.getAsJsonPrimitive("expand").getAsBoolean();
  }
  
  /**
   * Copy condense tensor.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @return the tensor
   */
  @Nonnull
  public static Tensor copyCondense(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] >= outDim[0];
    assert inDim[1] >= outDim[1];
    assert inDim[2] < outDim[2];
    assert 0 == inDim[0] % outDim[0];
    assert 0 == inDim[1] % outDim[1];
    final int kernelSizeX = inDim[0] / outDim[0];
    final int kernelSizeY = inDim[0] / outDim[0];
    int index = 0;
    @Nullable final double[] outputDataData = outputData.getData();
    for (int z = 0; z < inDim[2]; z++) {
      for (int xx = 0; xx < kernelSizeX; xx++) {
        for (int yy = 0; yy < kernelSizeY; yy++) {
          for (int y = 0; y < inDim[1]; y += kernelSizeY) {
            for (int x = 0; x < inDim[0]; x += kernelSizeX) {
              outputDataData[index++] = inputData.get(x + xx, y + yy, z);
            }
          }
        }
      }
    }
    return outputData;
  }
  
  /**
   * Copy expand tensor.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @return the tensor
   */
  @Nonnull
  public static Tensor copyExpand(@Nonnull final Tensor inputData, @Nonnull final Tensor outputData) {
    @Nonnull final int[] inDim = inputData.getDimensions();
    @Nonnull final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] <= outDim[0];
    assert inDim[1] <= outDim[1];
    assert inDim[2] > outDim[2];
    assert 0 == outDim[0] % inDim[0];
    assert 0 == outDim[1] % inDim[1];
    final int kernelSizeX = outDim[0] / inDim[0];
    final int kernelSizeY = outDim[0] / inDim[0];
    int index = 0;
    for (int z = 0; z < outDim[2]; z++) {
      for (int xx = 0; xx < kernelSizeX; xx++) {
        for (int yy = 0; yy < kernelSizeY; yy++) {
          for (int y = 0; y < outDim[1]; y += kernelSizeY) {
            for (int x = 0; x < outDim[0]; x += kernelSizeX) {
              outputData.set(x + xx, y + yy, z, inputData.getData()[index++]);
            }
          }
        }
      }
    }
    return outputData;
  }
  
  /**
   * From json img reshapeCast layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img reshapeCast layer
   */
  public static ImgReshapeLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgReshapeLayer(json);
  }
  
  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input-> input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    
    final Result input = inObj[0];
    final TensorList batch = input.getData();
    @Nonnull final int[] inputDims = batch.getDimensions();
    assert 3 == inputDims.length;
    assert expand || 0 == inputDims[0] % kernelSizeX : (inputDims[0] + " % " + kernelSizeX);
    assert expand || 0 == inputDims[1] % kernelSizeY : (inputDims[1] + " % " + kernelSizeY);
    assert !expand || 0 == inputDims[2] % (kernelSizeX * kernelSizeY);
    //assert input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    Tensor outputDims;
    if (expand) {
      outputDims = new Tensor(inputDims[0] * kernelSizeX,
        inputDims[1] * kernelSizeY,
        inputDims[2] / (kernelSizeX * kernelSizeY));
    }
    else {
      outputDims = new Tensor(inputDims[0] / kernelSizeX,
        inputDims[1] / kernelSizeY,
        inputDims[2] * kernelSizeX * kernelSizeY);
    }
    TensorArray data = TensorArray.wrap(IntStream.range(0, batch.length()).parallel()
      .mapToObj(dataIndex -> {
        Tensor inputData = batch.get(dataIndex);
        Tensor tensor = expand ? ImgReshapeLayer.copyExpand(inputData, outputDims.copy()) : ImgReshapeLayer.copyCondense(inputData, outputDims.copy());
        inputData.freeRef();
        return tensor;
      })
      .toArray(i -> new Tensor[i]));
    outputDims.freeRef();
    return new Result(data, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList error) -> {
      //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
      if (input.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, error.length()).parallel()
          .mapToObj(dataIndex -> {
            @Nonnull final Tensor passback = new Tensor(inputDims);
            @Nullable final Tensor err = error.get(dataIndex);
            Tensor tensor = expand ? ImgReshapeLayer.copyCondense(err, passback) : ImgReshapeLayer.copyExpand(err, passback);
            err.freeRef();
            return tensor;
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
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("kernelSizeX", kernelSizeX);
    json.addProperty("kernelSizeY", kernelSizeX);
    json.addProperty("expand", expand);
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
