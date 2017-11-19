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
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Img reshape layer.
 */
public class ImgReshapeLayer extends NNLayer {
  
  
  private final int kernelSizeX;
  private final int kernelSizeY;
  private final boolean expand;
  
  /**
   * Instantiates a new Img reshape layer.
   *
   * @param json the json
   */
  protected ImgReshapeLayer(JsonObject json) {
    super(json);
    this.kernelSizeX = json.getAsJsonPrimitive("kernelSizeX").getAsInt();
    this.kernelSizeY = json.getAsJsonPrimitive("kernelSizeY").getAsInt();
    this.expand = json.getAsJsonPrimitive("expand").getAsBoolean();
  }
  
  /**
   * Instantiates a new Img reshape layer.
   *
   * @param kernelSizeX the kernel size x
   * @param kernelSizeY the kernel size y
   * @param expand      the expand
   */
  public ImgReshapeLayer(int kernelSizeX, int kernelSizeY, boolean expand) {
    super();
    this.kernelSizeX = kernelSizeX;
    this.kernelSizeY = kernelSizeY;
    this.expand = expand;
  }
  
  /**
   * From json img reshape layer.
   *
   * @param json the json
   * @return the img reshape layer
   */
  public static ImgReshapeLayer fromJson(JsonObject json) {
    return new ImgReshapeLayer(json);
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
    assert inDim[0] >= outDim[0];
    assert inDim[1] >= outDim[1];
    assert inDim[2] < outDim[2];
    assert 0 == inDim[0] % outDim[0];
    assert 0 == inDim[1] % outDim[1];
    int kernelSizeX = inDim[0] / outDim[0];
    int kernelSizeY = inDim[0] / outDim[0];
    int index = 0;
    double[] outputDataData = outputData.getData();
    for (int xx = 0; xx < kernelSizeX; xx++) {
      for (int yy = 0; yy < kernelSizeY; yy++) {
        for (int z = 0; z < inDim[2]; z++) {
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
  public static Tensor copyExpand(Tensor inputData, Tensor outputData) {
    int[] inDim = inputData.getDimensions();
    int[] outDim = outputData.getDimensions();
    assert 3 == inDim.length;
    assert 3 == outDim.length;
    assert inDim[0] <= outDim[0];
    assert inDim[1] <= outDim[1];
    assert inDim[2] > outDim[2];
    assert 0 == outDim[0] % inDim[0];
    assert 0 == outDim[1] % inDim[1];
    int kernelSizeX = outDim[0] / inDim[0];
    int kernelSizeY = outDim[0] / inDim[0];
    int index = 0;
    for (int xx = 0; xx < kernelSizeX; xx++) {
      for (int yy = 0; yy < kernelSizeY; yy++) {
        for (int z = 0; z < outDim[2]; z++) {
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
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("kernelSizeX", kernelSizeX);
    json.addProperty("kernelSizeY", kernelSizeX);
    json.addProperty("expand", expand);
    return json;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    //assert Arrays.stream(inObj).flatMapToDouble(input-> input.getData().stream().flatMapToDouble(x-> Arrays.stream(x.getData()))).allMatch(v->Double.isFinite(v));
    
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    final int[] inputDims = batch.get(0).getDimensions();
    assert (3 == inputDims.length);
    assert (expand || 0 == inputDims[0] % kernelSizeX);
    assert (expand || 0 == inputDims[1] % kernelSizeX);
    assert (!expand || 0 == inputDims[2] % (kernelSizeX * kernelSizeY));
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
    return new NNResult(IntStream.range(0, batch.length()).parallel()
      .mapToObj(dataIndex -> expand ? copyExpand(batch.get(dataIndex), outputDims.copy()) : copyCondense(batch.get(dataIndex), outputDims.copy()))
      .toArray(i -> new Tensor[i])) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList error) {
        //assert error.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        if (input.isAlive()) {
          final Tensor[] data1 = IntStream.range(0, error.length()).parallel()
            .mapToObj(dataIndex -> {
              Tensor passback = new Tensor(inputDims);
              Tensor err = error.get(dataIndex);
              return expand ? copyCondense(err, passback) : copyExpand(err, passback);
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
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
