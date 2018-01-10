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
          int kernelBand = getBand(leg, convolutionParams, coords);
          return kernel.get(coords[0], coords[1], kernelBand);
        }));
      }
    }
    return this;
  }
  
  public Tensor extractKernel() {
    return extract(l -> l.writeKernel());
  }
  
  public Tensor extractDelta(DeltaSet<NNLayer> deltaSet, boolean remove) {
    return extract(l -> l.readDelta(deltaSet, remove));
  }
  
  public Tensor extract(Function<ExplodedConvolutionLeg, Tensor> extractor) {
    if (1 == subLayers.size()) {
      return extractor.apply(subLayers.get(0));
    }
    else {
      final Tensor filterDelta = new Tensor(convolutionParams.masterFilterDimensions);
      for (int legNumber = 0; legNumber < subLayers.size(); legNumber++) {
        final ExplodedConvolutionLeg leg = subLayers.get(legNumber);
        extractor.apply(leg).forEach((v, c) -> {
          int[] coords = c.getCoords();
          int kernelBand = getBand(leg, convolutionParams, coords);
          filterDelta.set(coords[0], coords[1], kernelBand, v);
        }, false);
      }
      return filterDelta;
    }
  }
  
  private int getBand(ExplodedConvolutionLeg leg, ConvolutionParams convolutionParams, int[] coords) {
    int legKernelBand = coords[2];
    int legKernelBandI;
    int legKernelBandO;
    if (false) {
      legKernelBandI = legKernelBand % (leg.toBand - leg.fromBand);
      legKernelBandO = (legKernelBand - legKernelBandI) / (leg.toBand - leg.fromBand);
    }
    else {
      legKernelBandO = legKernelBand % (convolutionParams.outputBands);
      legKernelBandI = (legKernelBand - legKernelBandO) / (convolutionParams.outputBands);
    }
    int offsetI = leg.fromBand;
    int filterBand;
    if (false) {
      filterBand = (offsetI + legKernelBandI) * convolutionParams.outputBands + legKernelBandO;
    }
    else {
      filterBand = (offsetI + legKernelBandI) + convolutionParams.inputBands * legKernelBandO;
    }
    return filterBand;
  }
  
  public PipelineNetwork getNetwork() {
    if (1 == subLayers.size()) {
      return subLayers.get(0).getNetwork();
    }
    else {
      PipelineNetwork network = new PipelineNetwork(1);
      DAGNode input = network.getInput(0);
      network.add(new BinarySumLayer(),
                  subLayers.stream().map(l -> {
                    return l.buildNetwork(network, network.add(new ImgBandSelectLayer(l.fromBand, l.toBand), input));
                  }).toArray(i -> new DAGNode[i]));
      return network;
    }
  }
  
}
