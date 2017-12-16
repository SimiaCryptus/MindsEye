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

package com.simiacryptus.mindseye.network.util;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;

/**
 * The type Polynomial convolution network.
 */
public class PolynomialConvolutionNetwork extends PolynomialNetwork {
  
  private final int radius;
  private final boolean simple;
  
  /**
   * Instantiates a new Polynomial convolution network.
   *
   * @param json the json
   */
  protected PolynomialConvolutionNetwork(JsonObject json) {
    super(json);
    radius = json.get("radius").getAsInt();
    simple = json.get("simple").getAsBoolean();
  }
  
  /**
   * Instantiates a new Polynomial convolution network.
   *
   * @param inputDims  the input dims
   * @param outputDims the output dims
   * @param radius     the radius
   * @param simple     the simple
   */
  public PolynomialConvolutionNetwork(int[] inputDims, int[] outputDims, int radius, boolean simple) {
    super(inputDims, outputDims);
    this.radius = radius;
    this.simple = simple;
  }
  
  /**
   * From json polynomial convolution network.
   *
   * @param json the json
   * @return the polynomial convolution network
   */
  public static PolynomialConvolutionNetwork fromJson(JsonObject json) {
    return new PolynomialConvolutionNetwork(json);
  }
  
  @Override
  public NNLayer newBias(int[] dims, double weight) {
    return new ImgBandBiasLayer(dims[2]).setWeights(i -> weight);
  }
  
  @Override
  public NNLayer newProductLayer() {
    return new ProductLayer();
  }
  
  @Override
  public NNLayer newSynapse(double weight) {
    return new ConvolutionLayer(radius, radius, inputDims[2] * outputDims[2]).set(i -> weight * (Math.random() - 0.5));
  }
}
