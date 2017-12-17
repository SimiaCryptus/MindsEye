/*
 * Copyright (c) 2017 by Andrew Charneski.
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Img crop layer.
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
  protected ImgCropLayer(final JsonObject json) {
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
  public static Tensor copyCondense(final Tensor inputData, final Tensor outputData) {
    final int[] inDim = inputData.getDimensions();
    final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2] : Arrays.toString(inDim) + "; " + Arrays.toString(outDim);
    assert inDim[0] >= outDim[0] : Arrays.toString(inDim) + "; " + Arrays.toString(outDim);
    assert inDim[1] >= outDim[1] : Arrays.toString(inDim) + "; " + Arrays.toString(outDim);
    final int paddingX = (inDim[0] - outDim[0]) / 2;
    final int paddingY = (inDim[0] - outDim[0]) / 2;
    outputData.coordStream().forEach((c) ->
      outputData.set(c, inputData.get(c.getCoords()[0] + paddingX, c.getCoords()[1] + paddingY, c.getCoords()[2])));
    return outputData;
  }
  
  /**
   * Copy expand tensor.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @return the tensor
   */
  public static Tensor copyExpand(final Tensor inputData, final Tensor outputData) {
    final int[] inDim = inputData.getDimensions();
    final int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] <= outDim[0];
    assert inDim[1] <= outDim[1];
    assert inDim[2] == outDim[2];
    final int paddingX = (outDim[0] - inDim[0]) / 2;
    final int paddingY = (outDim[0] - inDim[0]) / 2;
    inputData.coordStream().forEach(c -> {
      outputData.set(c.getCoords()[0] + paddingX, c.getCoords()[1] + paddingY, c.getCoords()[2], inputData.get(c));
    });
    return outputData;
  }
  
  /**
   * From json img crop layer.
   *
   * @param json the json
   * @return the img crop layer
   */
  public static ImgCropLayer fromJson(final JsonObject json) {
    return new ImgCropLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    assert Arrays.stream(inObj).flatMapToDouble(input -> input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData()))).allMatch(v -> Double.isFinite(v));
    
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputDims = batch.get(0).getDimensions();
    assert 3 == inputDims.length;
    assert input.getData().stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    return new NNResult(IntStream.range(0, batch.length()).parallel()
      .mapToObj(dataIndex -> {
        final Tensor outputDims = new Tensor(sizeX, sizeY, inputDims[2]);
        return ImgCropLayer.copyCondense(batch.get(dataIndex), outputDims);
      })
      .toArray(i -> new Tensor[i])) {
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList error) {
        assert error.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        if (input.isAlive()) {
          final Tensor[] data1 = IntStream.range(0, error.length()).parallel()
            .mapToObj(dataIndex -> {
              final Tensor err = error.get(dataIndex);
              final Tensor passback = new Tensor(inputDims);
              return ImgCropLayer.copyExpand(err, passback);
            }).toArray(i -> new Tensor[i]);
          input.accumulate(buffer, new TensorArray(data1));
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeX);
    return json;
  }
  
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
