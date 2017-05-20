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

package com.simiacryptus.mindseye.graph;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.graph.dag.DAGNetwork;
import com.simiacryptus.mindseye.graph.dag.DAGNode;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.media.ImgConvolutionSynapseLayer;
import com.simiacryptus.mindseye.net.reducers.ImgConcatLayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * See Also http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf
 */
public class InceptionLayer extends DAGNetwork {
  
  private final HashMap<NNLayer, NNLayer> forwardLinkIndex = new HashMap<>();
  private final HashMap<NNLayer, NNLayer> backwardLinkIndex = new HashMap<>();
  public final int[][][] kernels;
  private final DAGNode head;
  
  public InceptionLayer(int[][][] kernels) {
    super(1);
    this.kernels = kernels;
    List<DAGNode> kernelLayers = new ArrayList<>();
    for(int[][] kernelPipeline : this.kernels) {
      PipelineNetwork kernelPipelineNetwork = new PipelineNetwork();
      for(int[] kernel : kernelPipeline) {
        kernelPipelineNetwork.add(new ImgConvolutionSynapseLayer(kernel[0], kernel[1], kernel[2]));
      }
      kernelLayers.add(add(kernelPipelineNetwork, getInput(0)));
    }
    assert (0 < kernelLayers.size());
    this.head = add(new ImgConcatLayer(), kernelLayers.toArray(new DAGNode[]{}));
  }
  
  @Override
  public DAGNode getHead() {
    assert null != this.head;
    return this.head;
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.add("root", getHead().toJson());
    return json;
  }
  
}
