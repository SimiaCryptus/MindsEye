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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.ImgConcatLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;

/**
 * The type Inception layer.
 */
@SuppressWarnings("serial")
public class InceptionLayer extends DAGNetwork {
  
  /**
   * The Kernels.
   */
  public final int[][][] kernels;
  private final DAGNode head;
  /**
   * The Convolution layers.
   */
  @Nonnull
  List<ConvolutionLayer> convolutionLayers = new ArrayList<>();
  
  /**
   * Instantiates a new Inception layer.
   *
   * @param kernels the kernels
   */
  public InceptionLayer(final int[][][] kernels) {
    super(1);
    this.kernels = kernels;
    @Nonnull final List<DAGNode> pipelines = new ArrayList<>();
    for (@Nonnull final int[][] kernelPipeline : this.kernels) {
      @Nonnull final PipelineNetwork kernelPipelineNetwork = new PipelineNetwork();
      for (final int[] kernel : kernelPipeline) {
        @Nonnull final ConvolutionLayer convolutionSynapseLayer = new ConvolutionLayer(kernel[0], kernel[1], kernel[2]);
        convolutionLayers.add(convolutionSynapseLayer);
        kernelPipelineNetwork.add(convolutionSynapseLayer);
      }
      pipelines.add(add(kernelPipelineNetwork, getInput(0)));
    }
    assert 0 < pipelines.size();
    head = add(new ImgConcatLayer(), pipelines.toArray(new DAGNode[]{}));
  }
  
  @Override
  public DAGNode getHead() {
    assert null != head;
    return head;
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJson(resources, dataSerializer);
    json.add("root", getHead().getLayer().getJson(resources, dataSerializer));
    return json;
  }
  
  /**
   * Sets weights.
   *
   * @param f the f
   * @return the weights
   */
  @Nonnull
  public InceptionLayer setWeights(@Nonnull final DoubleSupplier f) {
    convolutionLayers.forEach(x -> x.setWeights(f));
    return this;
  }
  
}
