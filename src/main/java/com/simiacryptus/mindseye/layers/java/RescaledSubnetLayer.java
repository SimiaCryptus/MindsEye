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
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.NNResult;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * This layer works as a scaling function, similar to a father wavelet. Allows convolutional and pooling layers to work
 * across larger image regions.
 */
@SuppressWarnings("serial")
public class RescaledSubnetLayer extends NNLayer {
  
  private final int scale;
  @Nullable
  private final NNLayer subnetwork;
  
  /**
   * Instantiates a new Rescaled subnet layer.
   *
   * @param scale      the scale
   * @param subnetwork the subnetwork
   */
  public RescaledSubnetLayer(final int scale, final NNLayer subnetwork) {
    super();
    this.scale = scale;
    this.subnetwork = subnetwork;
    this.subnetwork.addRef();
  }
  
  /**
   * Instantiates a new Rescaled subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected RescaledSubnetLayer(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    scale = json.getAsJsonPrimitive("scale").getAsInt();
    JsonObject subnetwork = json.getAsJsonObject("subnetwork");
    this.subnetwork = subnetwork == null ? null : NNLayer.fromJson(subnetwork, rs);
  }
  
  /**
   * From json rescaled subnet layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the rescaled subnet layer
   */
  public static RescaledSubnetLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new RescaledSubnetLayer(json, rs);
  }
  
  @Override
  protected void _free() {
    this.subnetwork.freeRef();
    super._free();
  }
  
  @javax.annotation.Nullable
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    assert 1 == inObj.length;
    final NNResult input = inObj[0];
    final TensorList batch = input.getData();
    @javax.annotation.Nonnull final int[] inputDims = batch.get(0).getDimensions();
    assert 3 == inputDims.length;
    if (1 == scale) return subnetwork.eval(inObj);
    
    @javax.annotation.Nonnull final PipelineNetwork network = new PipelineNetwork();
    @javax.annotation.Nullable final DAGNode condensed = network.wrap(new ImgReshapeLayer(scale, scale, false));
    network.wrap(new ImgConcatLayer(), IntStream.range(0, scale * scale).mapToObj(subband -> {
      @javax.annotation.Nonnull final int[] select = new int[inputDims[2]];
      for (int i = 0; i < inputDims[2]; i++) {
        select[i] = subband * inputDims[2] + i;
      }
      return network.add(subnetwork, network.wrap(new ImgBandSelectLayer(select), condensed));
    }).toArray(i -> new DAGNode[i]));
    network.wrap(new ImgReshapeLayer(scale, scale, true));
    
    NNResult eval = network.eval(inObj);
    network.freeRef();
    return eval;
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("scale", scale);
    json.add("subnetwork", subnetwork.getJson(resources, dataSerializer));
    return json;
  }
  
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  
}
