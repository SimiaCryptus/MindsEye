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
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.layers.cudnn.ImgTileSelectLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * This layer works as a scaling function, similar to a father wavelet. Allows convolutional and pooling layers to work
 * across larger image regions.
 */
@SuppressWarnings("serial")
public class ImgTileSubnetLayer extends LayerBase {
  
  private final int height;
  private final int width;
  private final int strideX;
  private final int strideY;
  @Nullable
  private final Layer subnetwork;
  
  /**
   * Instantiates a new Rescaled subnet layer.
   *
   * @param height     the scale
   * @param subnetwork the subnetwork
   */
  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height, final int strideX, final int strideY) {
    super();
    this.height = height;
    this.width = width;
    this.strideX = strideX;
    this.strideY = strideY;
    this.subnetwork = subnetwork;
    this.subnetwork.addRef();
  }
  
  public ImgTileSubnetLayer(final Layer subnetwork, final int width, final int height) {
    this(subnetwork, width, height, width, height);
  }
  
  /**
   * Instantiates a new Rescaled subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ImgTileSubnetLayer(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    height = json.getAsJsonPrimitive("height").getAsInt();
    width = json.getAsJsonPrimitive("width").getAsInt();
    strideX = json.getAsJsonPrimitive("strideX").getAsInt();
    strideY = json.getAsJsonPrimitive("strideY").getAsInt();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
    this.subnetwork = subnetwork == null ? null : Layer.fromJson(subnetwork, rs);
  }
  
  /**
   * From json rescaled subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the rescaled subnet layer
   */
  public static ImgTileSubnetLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgTileSubnetLayer(json, rs);
  }
  
  @Override
  protected void _free() {
    this.subnetwork.freeRef();
    super._free();
  }
  
  @Nullable
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    @javax.annotation.Nonnull final int[] inputDims = inObj[0].getData().getDimensions();
    assert 3 == inputDims.length;
    @javax.annotation.Nonnull final PipelineNetwork network = new PipelineNetwork();
    int cols = (int) (Math.ceil((inputDims[0] - width) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((inputDims[1] - height) * 1.0 / strideY) + 1);
    if (cols == 1 && rows == 1) return subnetwork.eval(inObj);
    DAGNode input = network.getInput(0);
    ArrayList<DAGNode> nodes = new ArrayList<>();
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        int positionX = col * strideX;
        int positionY = row * strideY;
        assert positionX >= 0;
        assert positionY >= 0;
        assert positionX < inputDims[0];
        assert positionY < inputDims[1];
        nodes.add(
          network.add(subnetwork,
            network.wrap(
              new ImgTileSelectLayer(width, height, positionX, positionY),
              input))
        );
      }
    }
    network.wrap(new ImgTileAssemblyLayer(cols, rows), nodes.toArray(new DAGNode[]{}));
    Result eval = network.eval(inObj);
    network.freeRef();
    return eval;
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("height", height);
    json.addProperty("width", width);
    json.addProperty("strideX", strideX);
    json.addProperty("strideY", strideY);
    json.add("subnetwork", subnetwork.getJson(resources, dataSerializer));
    return json;
  }
  
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
