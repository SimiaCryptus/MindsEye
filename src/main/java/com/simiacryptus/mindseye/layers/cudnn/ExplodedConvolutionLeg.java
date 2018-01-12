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
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

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
    int outputBands = this.convolutionParams.outputBands;
    filterDimensions[2] = inputBands * outputBands;
    assert Arrays.equals(kernel.getDimensions(), filterDimensions) : Arrays.toString(kernel.getDimensions()) + " != " + Arrays.toString(filterDimensions);
    final int inputBandsSq = inputBands * inputBands;
    for (int layerNumber = 0; layerNumber < subLayers.size(); layerNumber++) {
      final int filterBandOffset = layerNumber * inputBandsSq;
      subLayers.get(layerNumber).set(new Tensor(filterDimensions[0], filterDimensions[1], inputBandsSq).setByCoord(c -> {
        int[] coords = c.getCoords();
        int filterBand = getFilterBand(filterBandOffset, coords[2]);
        if (filterBand < filterDimensions[2]) {
          return kernel.get(coords[0], coords[1], filterBand);
        }
        else {
          return 0;
        }
      }));
    }
    return this;
  }
  
  public Tensor read(Function<SimpleConvolutionLayer, Tensor> extractor) {
    int inputBands = getInputBands();
    final int[] filterDimensions = Arrays.copyOf(this.convolutionParams.masterFilterDimensions, this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    int outputBands = convolutionParams.outputBands;
    Tensor resultDelta = new Tensor(filterDimensions[0], filterDimensions[1], inputBands * outputBands);
    for (int layerNumber = 0; layerNumber < subLayers.size(); layerNumber++) {
      final int filterBandOffset = layerNumber * inputBands * inputBands;
      Tensor deltaTensor = extractor.apply(subLayers.get(layerNumber));
      if (null != deltaTensor) {
        deltaTensor.forEach((v, c) -> {
          int[] coords = c.getCoords();
          int filterBand = getFilterBand(filterBandOffset, coords[2]);
          if (filterBand < filterDimensions[2]) {
            resultDelta.set(coords[0], coords[1], filterBand, v);
          }
        }, false);
      }
    }
    return resultDelta;
  }
  
  public int getFilterBand(int filterBandOffset, int filterBand) {
    filterBand = filterBandOffset + filterBand;
    filterBand = ConvolutionLayer.transposeCoordinates(getInputBands(), convolutionParams.outputBands, filterBand);
    return filterBand;
  }
  
  public int getInputBands() {
    return this.toBand - this.fromBand;
  }
  
  public Tensor read(DeltaSet<NNLayer> deltaSet, boolean remove) {
    return read((sublayer) -> {
      final Delta<NNLayer> subnetDelta = remove ? deltaSet.getMap().remove(sublayer) : deltaSet.getMap().get(sublayer);
      if (null == subnetDelta) throw new RuntimeException("No Delta for " + sublayer);
      return new Tensor(subnetDelta.getDelta(), sublayer.kernel.getDimensions());
    });
  }
  
  public Tensor read() {
    return read((sublayer) -> {
      return sublayer.kernel;
    });
  }
  
  public DAGNode add(final DAGNode input) {
    DAGNetwork network = input.getNetwork();
    DAGNode head = input;
    final int[] filterDimensions = this.convolutionParams.masterFilterDimensions;
    if (getInputBands() == this.convolutionParams.outputBands) {
      assert 1 == subLayers.size();
      head = network.add(subLayers.get(0), head);
    }
    else {
      head = network.add(new ImgConcatLayer().setMaxBands(this.convolutionParams.outputBands).setPrecision(this.convolutionParams.precision),
                         subLayers.stream().map(l -> {
                           return network.add(l, input);
                         }).toArray(i -> new DAGNode[i]));
    }
    if (this.convolutionParams.paddingX != null || this.convolutionParams.paddingY != null) {
      int x = ((filterDimensions[0] - 1) / 2);
      if (this.convolutionParams.paddingX != null) x = this.convolutionParams.paddingX - x;
      int y = ((filterDimensions[1] - 1) / 2);
      if (this.convolutionParams.paddingY != null) y = this.convolutionParams.paddingY - y;
      head = network.add(new ImgZeroPaddingLayer(x, y), head);
    }
    return head;
  }
}
