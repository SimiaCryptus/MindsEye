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

package com.simiacryptus.mindseye.layers.media;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.ml.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Img reshape layer.
 */
public class ImgCropLayer extends NNLayer {
  
  
  private final int sizeX;
  private final int sizeY;
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeX);
    return json;
  }
  
  /**
   * From json img reshape layer.
   *
   * @param json the json
   * @return the img reshape layer
   */
  public static ImgCropLayer fromJson(JsonObject json) {
    return new ImgCropLayer(json);
  }

  /**
   * Instantiates a new Img reshape layer.
   *
   * @param json the json
   */
  protected ImgCropLayer(JsonObject json) {
    super(json);
    this.sizeX = json.getAsJsonPrimitive("sizeX").getAsInt();
    this.sizeY = json.getAsJsonPrimitive("sizeY").getAsInt();
  }
  
  /**
   * Instantiates a new image cropping layer.
   *
   * @param sizeX the kernel size x
   * @param sizeY the kernel size y
   */
  public ImgCropLayer(int sizeX, int sizeY) {
    super();
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    assert Arrays.stream(inObj).flatMapToDouble(input->input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    
    final NNResult input = inObj[0];
    final TensorList batch = input.data;
    final int[] inputDims = batch.get(0).getDimensions();
    assert(3 == inputDims.length);
    assert input.data.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
    Tensor outputDims = new Tensor(inputDims[0] - sizeX,
                                    inputDims[1] - sizeY,
                                    inputDims[2]);
    return new NNResult(IntStream.range(0, batch.length()).parallel()
                           .mapToObj(dataIndex -> copyCondense(batch.get(dataIndex), outputDims))
                           .toArray(i -> new Tensor[i])) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList error) {
        assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        if (input.isAlive()) {
          final Tensor[] data1 = IntStream.range(0, error.length()).parallel()
             .mapToObj(dataIndex -> {
               Tensor passback = new Tensor(inputDims);
               Tensor err = error.get(dataIndex);
               return copyExpand(err, passback);
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
  
  /**
   * Copy condense tensor.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @return the tensor
   */
  public static Tensor copyCondense(Tensor inputData, Tensor outputData) {
    int[] inDim = inputData.getDimensions();
    int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[2] == outDim[2];
    assert inDim[0] >= outDim[0];
    assert inDim[1] >= outDim[1];
    int paddingX = (inDim[0] - outDim[0]);
    int paddingY = (inDim[0] - outDim[0]);
    outputData.coordStream(true).forEach((c)->outputData.set(c,inputData.get(c.coords[0] + paddingX,c.coords[1] + paddingY,c.coords[2])));
    return outputData;
  }
  
  /**
   * Copy expand tensor.
   *
   * @param inputData  the input data
   * @param outputData the output data
   * @return the tensor
   */
  public static Tensor copyExpand(Tensor inputData, Tensor outputData) {
    int[] inDim = inputData.getDimensions();
    int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] <= outDim[0];
    assert inDim[1] <= outDim[1];
    assert inDim[2] == outDim[2];
    int paddingX = (outDim[0] - inDim[0]);
    int paddingY = (outDim[0] - inDim[0]);
    outputData.coordStream().forEach(c->{
      int x = c.coords[0] - paddingX;
      int y = c.coords[1] - paddingY;
      if(x >= 0 && x < outDim[0] && y >= 0 && y < outDim[1]) {
        outputData.set(c, inputData.get(x,y,c.coords[2]));
      }
    });
    return outputData;
  }
  
  
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
