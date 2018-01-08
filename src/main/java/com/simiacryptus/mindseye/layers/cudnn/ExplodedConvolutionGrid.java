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

package com.simiacryptus.mindseye.layers.cudnn;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import java.util.ArrayList;
import java.util.List;

public class ExplodedConvolutionGrid {
  /**
   * The Sub layers.
   */
  public final List<ExplodedConvolutionLeg> subLayers;
  public final ConvolutionParams convolutionParams;
  
  /**
   * Instantiates a new Exploded network.
   */
  public ExplodedConvolutionGrid(ConvolutionParams convolutionParams, int maxBandBatch) {
    this.convolutionParams = convolutionParams;
    subLayers = new ArrayList<>();
    for (int inputBandOffset = 0; inputBandOffset < convolutionParams.inputBands; inputBandOffset += maxBandBatch) {
      int toBand = Math.min(convolutionParams.inputBands, inputBandOffset + maxBandBatch);
      int kernelBandOffset = inputBandOffset * convolutionParams.outputBands;
      int kernelBandSkip = 1;
      subLayers.add(new ExplodedConvolutionLeg(convolutionParams, inputBandOffset, toBand, kernelBandOffset, kernelBandSkip));
    }
  }
  
  public ExplodedConvolutionGrid write(Tensor kernel) {
    for (ExplodedConvolutionLeg leg : subLayers) {
      leg.write(kernel);
    }
    return this;
  }
  
  public Tensor extractDelta(DeltaSet<NNLayer> deltaSet, boolean remove) {
    final Tensor filterDelta = new Tensor(convolutionParams.filterDimensions);
    for (int legNumber = 0; legNumber < subLayers.size(); legNumber++) {
      final ExplodedConvolutionLeg leg = subLayers.get(legNumber);
      leg.extractDelta(deltaSet, filterDelta, remove);
    }
    return filterDelta;
  }
  
  public PipelineNetwork getNetwork() {
    PipelineNetwork network = new PipelineNetwork(1);
    if (convolutionParams.inputBands != convolutionParams.outputBands) {
      network.add(new BinarySumLayer(),
                  subLayers.stream().map(l -> {
                    return network.add(l.getNetwork(), network.getInput(0));
                  }).toArray(i -> new DAGNode[i]));
    }
    else {
      assert 1 == subLayers.size();
      network.add(subLayers.get(0).getNetwork());
    }
    return network;
  }
  
}
