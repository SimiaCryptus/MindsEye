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

package com.simiacryptus.mindseye.network.util;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;

import java.util.Map;

/**
 * The type Polynomial convolution network.
 */
@SuppressWarnings("serial")
public class PolynomialConvolutionNetwork extends PolynomialNetwork {
  
  private final int radius;
  
  /**
   * Instantiates a new Polynomial convolution network.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   * @param radius     the radius
   * @param simple     the simple
   */
  public PolynomialConvolutionNetwork(final int[] inputDims, final int[] outputDims, final int radius, final boolean simple) {
    super(inputDims, outputDims);
    this.radius = radius;
  }
  
  /**
   * Instantiates a new Polynomial convolution network.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected PolynomialConvolutionNetwork(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    super(json, rs);
    radius = json.get("radius").getAsInt();
    json.get("simple").getAsBoolean();
  }
  
  /**
   * From json polynomial convolution network.
   *
   * @param json the json
   * @param rs   the rs
   * @return the polynomial convolution network
   */
  public static PolynomialConvolutionNetwork fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new PolynomialConvolutionNetwork(json, rs);
  }
  
  @javax.annotation.Nonnull
  @Override
  public Layer newBias(final int[] dims, final double weight) {
    return new ImgBandBiasLayer(dims[2]).setWeights(i -> weight);
  }
  
  @javax.annotation.Nonnull
  @Override
  public Layer newProductLayer() {
    return new ProductLayer();
  }
  
  @javax.annotation.Nonnull
  @Override
  public Layer newSynapse(final double weight) {
    return new ConvolutionLayer(radius, radius, inputDims[2], outputDims[2]).set(i -> weight * (Math.random() - 0.5));
  }
}
