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
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.MultiPrecision;
import com.simiacryptus.util.io.JsonUtil;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * A dense matrix operator using vector-matrix multiplication. Represents a fully connected layer of synapses, where all
 * inputs are connected to all outputs via seperate coefficients.
 */
@SuppressWarnings("serial")
public class ReshapeLayer extends NNLayer implements MultiPrecision<ReshapeLayer> {
  private static final Logger log = LoggerFactory.getLogger(ReshapeLayer.class);
  /**
   * The Output dims.
   */
  public final @Nullable int[] outputDims;
  private Precision precision = Precision.Double;
  
  /**
   * Instantiates a new Img concat layer.
   */
  private ReshapeLayer() {
    outputDims = null;
  }
  
  /**
   * Instantiates a new Fully connected layer.
   *
   * @param outputDims the output dims
   */
  public ReshapeLayer(@javax.annotation.Nonnull final int... outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
  }
  
  /**
   * Instantiates a new Img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected ReshapeLayer(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    this.precision = Precision.valueOf(json.getAsJsonPrimitive("precision").getAsString());
  }
  
  /**
   * From json img concat layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img concat layer
   */
  public static ReshapeLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ReshapeLayer(json, rs);
  }
  
  @Override
  public @Nullable NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    assert 1 == inObj.length;
    TensorList data = inObj[0].getData();
    int[] inputDims = data.getDimensions();
    for (@javax.annotation.Nonnull NNResult nnResult : inObj) {
      nnResult.addRef();
    }
    return new NNResult(new ReshapedTensorList(data, outputDims), (DeltaSet<NNLayer> buffer, TensorList delta) -> {
      @javax.annotation.Nonnull ReshapedTensorList tensorList = new ReshapedTensorList(delta, inputDims);
      inObj[0].accumulate(buffer, tensorList);
      tensorList.freeRef();
    }) {
  
      @Override
      protected void _free() {
        for (@javax.annotation.Nonnull NNResult nnResult : inObj) {
          nnResult.freeRef();
        }
      }
  
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
    
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.addProperty("precision", precision.name());
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public Precision getPrecision() {
    return precision;
  }
  
  @javax.annotation.Nonnull
  @Override
  public ReshapeLayer setPrecision(final Precision precision) {
    this.precision = precision;
    return this;
  }
}
