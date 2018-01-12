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
import java.util.function.Function;

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
    for (int fromBand = 0; fromBand < convolutionParams.inputBands; fromBand += maxBandBatch) {
      int toBand = Math.min(convolutionParams.inputBands, fromBand + maxBandBatch);
      subLayers.add(new ExplodedConvolutionLeg(convolutionParams, fromBand, toBand));
    }
  }
  
  public ExplodedConvolutionGrid write(Tensor kernel) {
    if (1 == subLayers.size()) {
      subLayers.get(0).write(kernel);
    }
    else {
      for (ExplodedConvolutionLeg leg : subLayers) {
        int[] legDims = {convolutionParams.masterFilterDimensions[0], convolutionParams.masterFilterDimensions[1], (leg.toBand - leg.fromBand) * convolutionParams.outputBands};
        leg.write(new Tensor(legDims).mapCoords(c -> {
          int[] coords = c.getCoords();
          int kernelBand = getFilterBand(leg, coords[2]);
          return kernel.get(coords[0], coords[1], kernelBand);
        }));
      }
    }
    return this;
  }
  
  public Tensor read(Function<ExplodedConvolutionLeg, Tensor> extractor) {
    if (1 == subLayers.size()) {
      return extractor.apply(subLayers.get(0));
    }
    else {
      final Tensor filterDelta = new Tensor(convolutionParams.masterFilterDimensions);
      for (ExplodedConvolutionLeg leg : subLayers) {
        extractor.apply(leg).forEach((v, c) -> {
          int[] coords = c.getCoords();
          filterDelta.set(coords[0], coords[1], getFilterBand(leg, coords[2]), v);
        }, false);
      }
      return filterDelta;
    }
  }
  
  public Tensor read() {
    return read(l -> l.read());
  }
  
  public Tensor read(DeltaSet<NNLayer> deltaSet, boolean remove) {
    return read(l -> l.read(deltaSet, remove));
  }
  
  private int getFilterBand(ExplodedConvolutionLeg leg, int legFilterBand) {
    int filterBand = legFilterBand + leg.fromBand * convolutionParams.outputBands;
    //filterBand = ConvolutionLayer.transposeCoordinates(convolutionParams.inputBands, convolutionParams.outputBands, filterBand);
    return filterBand;
  }
  
  public PipelineNetwork getNetwork() {
    PipelineNetwork network = new PipelineNetwork(1);
    add(network.getInput(0));
    return network;
  }
  
  public DAGNode add(DAGNode input) {
    if (subLayers.size() == 1) {
      return subLayers.get(0).add(input);
    }
    DAGNode[] nodes = subLayers.stream().map(l -> {
      return l.add(input.getNetwork().add(new ImgBandSelectLayer(l.fromBand, l.toBand), input));
    }).toArray(i -> new DAGNode[i]);
    if (nodes.length > 1) return input.getNetwork().add(new BinarySumLayer(), nodes);
    return nodes[0];
  }
  
}
