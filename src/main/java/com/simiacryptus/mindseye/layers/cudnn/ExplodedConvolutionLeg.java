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

import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The type Exploded network.
 */
public class ExplodedConvolutionLeg {
  public final ConvolutionParams convolutionParams;
  public final List<SimpleConvolutionLayer> subLayers;
  public final int fromBand;
  public final int toBand;
  
  public ExplodedConvolutionLeg(ConvolutionParams convolutionParams, int fromBand, int toBand) {
    this.fromBand = fromBand;
    this.toBand = toBand;
    this.convolutionParams = convolutionParams;
    this.subLayers = new ArrayList<>();
    int inputBands = getInputBands();
    final int inputBandsSq = inputBands * inputBands;
    final int[] filterDimensions = Arrays.copyOf(this.convolutionParams.masterFilterDimensions, this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    for (int offset = 0; offset < filterDimensions[2]; offset += inputBandsSq) {
      final Tensor cellKernel = new Tensor(filterDimensions[0], filterDimensions[1], inputBandsSq);
      this.subLayers.add(new SimpleConvolutionLayer(cellKernel).setStrideX(this.convolutionParams.strideX) //
                                                               .setStrideY(this.convolutionParams.strideY) //
                                                               .setPrecision(this.convolutionParams.precision));
    }
  }
  
  
  public ExplodedConvolutionLeg write(Tensor kernel) {
    int inputBands = getInputBands();
    final int[] filterDimensions = Arrays.copyOf(this.convolutionParams.masterFilterDimensions, this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    final int inputBandsSq = inputBands * inputBands;
    for (int layerNumber = 0; layerNumber < subLayers.size(); layerNumber++) {
      final int bandOffset = layerNumber * inputBandsSq;
      subLayers.get(layerNumber).set(new Tensor(filterDimensions[0], filterDimensions[1], inputBandsSq).setByCoord(c -> {
        int[] coords = c.getCoords();
        int band = bandOffset + coords[2];
        band = ConvolutionLayer.transposeCoordinates(inputBands, this.convolutionParams.outputBands, band);
        if (band < filterDimensions[2]) {
          return kernel.get(coords[0], coords[1], band);
        }
        else {
          return 0;
        }
      }));
    }
    return this;
  }
  
  public int getInputBands() {
    return this.toBand - this.fromBand;
  }
  
  public Tensor extractDelta(DeltaSet<NNLayer> deltaSet, boolean remove) {
    int inputBands = getInputBands();
    int inputBandsSq = inputBands * inputBands;
    final int[] filterDimensions = Arrays.copyOf(this.convolutionParams.masterFilterDimensions, this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    int outputBands = convolutionParams.outputBands;
    Tensor resultDelta = new Tensor(filterDimensions[0], filterDimensions[1], inputBands * outputBands);
    for (int layerNumber = 0; layerNumber < subLayers.size(); layerNumber++) {
      final int bandOffset = layerNumber * inputBandsSq;
      final SimpleConvolutionLayer subLayer = subLayers.get(layerNumber);
      final Delta<NNLayer> subnetDelta = remove ? deltaSet.getMap().remove(subLayer) : deltaSet.getMap().get(subLayer);
      if (null != subnetDelta) {
        final Tensor deltaTensor = new Tensor(subnetDelta.getDelta(), subLayer.kernel.getDimensions());
        deltaTensor.forEach((v, c) -> {
          int[] coords = c.getCoords();
          int band = bandOffset + coords[2];
          band = ConvolutionLayer.transposeCoordinates(inputBands, outputBands, band);
          if (band < filterDimensions[2]) {
            resultDelta.set(coords[0], coords[1], band, v);
          }
        }, false);
      }
    }
    return resultDelta;
  }
  
  /**
   * The Network.
   */
  public PipelineNetwork getNetwork() {
    final int[] filterDimensions = this.convolutionParams.masterFilterDimensions;
    PipelineNetwork network = new PipelineNetwork(1);
    if (getInputBands() == this.convolutionParams.outputBands) {
      assert 1 == subLayers.size();
      network.add(subLayers.get(0));
    }
    else {
      DAGNode input = network.getHead();
      network.add(new ImgConcatLayer().setMaxBands(this.convolutionParams.outputBands).setPrecision(this.convolutionParams.precision),
                  subLayers.stream().map(l -> {
                    return network.add(l, input);
                  }).toArray(i -> new DAGNode[i]));
    }
    if (this.convolutionParams.paddingX != null || this.convolutionParams.paddingY != null) {
      int x = ((filterDimensions[0] - 1) / 2);
      if (this.convolutionParams.paddingX != null) x = this.convolutionParams.paddingX - x;
      int y = ((filterDimensions[1] - 1) / 2);
      if (this.convolutionParams.paddingY != null) y = this.convolutionParams.paddingY - y;
      network.add(new ImgZeroPaddingLayer(x, y));
    }
    return network;
  }
}
