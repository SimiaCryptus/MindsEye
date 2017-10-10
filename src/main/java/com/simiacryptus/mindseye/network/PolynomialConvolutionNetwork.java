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

package com.simiacryptus.mindseye.network;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.f32.ImgBandBiasLayer;

public class PolynomialConvolutionNetwork extends PolynomialNetwork {
  
  public static PolynomialConvolutionNetwork fromJson(JsonObject json) {
    return new PolynomialConvolutionNetwork(json);
  }
  
  private final int radius;
  private final boolean simple;
  
  protected PolynomialConvolutionNetwork(JsonObject json) {
    super(json);
    radius = json.get("radius").getAsInt();
    simple = json.get("simple").getAsBoolean();
  }
  
  public PolynomialConvolutionNetwork(int[] inputDims, int[] outputDims, int radius, boolean simple) {
    super(inputDims, outputDims);
    this.radius = radius;
    this.simple = simple;
  }
  
  @Override
  public NNLayer newBias(int[] dims, double weight) {
    return new ImgBandBiasLayer(dims[2]).setWeights(i->weight);
  }

  @Override
  public NNLayer newProductLayer() {
    return new com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer();
  }

  @Override
  public NNLayer newSynapse(double weight) {
    return new ConvolutionLayer(radius, radius, inputDims[2]*outputDims[2], simple).setWeights(i->weight*(Math.random()-0.5));
  }
}
