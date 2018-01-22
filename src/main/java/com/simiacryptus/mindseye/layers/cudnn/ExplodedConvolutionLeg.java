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

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 * The type Exploded network.
 */
class ExplodedConvolutionLeg {
  private static final Logger log = LoggerFactory.getLogger(ExplodedConvolutionLeg.class);
  
  /**
   * The Convolution params.
   */
  public final ConvolutionParams convolutionParams;
  /**
   * The Sub layers.
   */
  public final List<SimpleConvolutionLayer> subLayers;
  /**
   * The From band.
   */
  public final int fromBand;
  /**
   * The To band.
   */
  public final int toBand;
  
  /**
   * Instantiates a new Exploded convolution leg.
   *
   * @param convolutionParams the convolution params
   * @param fromBand          the from band
   * @param toBand            the to band
   */
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
  
  
  /**
   * Write exploded convolution leg.
   *
   * @param filter the kernel
   * @return the exploded convolution leg
   */
  public ExplodedConvolutionLeg write(Tensor filter) {
    int inputBands = getInputBands();
    final int[] filterDimensions = Arrays.copyOf(this.convolutionParams.masterFilterDimensions, this.convolutionParams.masterFilterDimensions.length);
    int outputBands = this.convolutionParams.outputBands;
    int squareOutputBands = (int) (Math.ceil(convolutionParams.outputBands * 1.0 / inputBands) * inputBands);
    assert squareOutputBands >= convolutionParams.outputBands : String.format("%d >= %d", squareOutputBands, convolutionParams.outputBands);
    assert squareOutputBands % inputBands == 0 : String.format("%d %% %d", squareOutputBands, inputBands);
    filterDimensions[2] = inputBands * outputBands;
    assert Arrays.equals(filter.getDimensions(), filterDimensions) : Arrays.toString(filter.getDimensions()) + " != " + Arrays.toString(filterDimensions);
    final int inputBandsSq = inputBands * inputBands;
    for (int layerNumber = 0; layerNumber < subLayers.size(); layerNumber++) {
      final int filterBandOffset = layerNumber * inputBandsSq;
      subLayers.get(layerNumber).set(new Tensor(filterDimensions[0], filterDimensions[1], inputBandsSq).setByCoord(c -> {
        int[] coords = c.getCoords();
        int filterBand = getFilterBand(filterBandOffset, coords[2], squareOutputBands);
        if (filterBand < filterDimensions[2]) {
          return filter.get(coords[0], coords[1], filterBand);
        }
        else {
          return 0;
        }
      }, false));
    }
    return this;
  }
  
  /**
   * Read tensor.
   *
   * @param extractor the extractor
   * @return the tensor
   */
  public Tensor read(Function<SimpleConvolutionLayer, Tensor> extractor) {
    int inputBands = getInputBands();
    final int[] filterDimensions = Arrays.copyOf(this.convolutionParams.masterFilterDimensions, this.convolutionParams.masterFilterDimensions.length);
    filterDimensions[2] = inputBands * this.convolutionParams.outputBands;
    int outputBands = convolutionParams.outputBands;
    int squareOutputBands = (int) (Math.ceil(convolutionParams.outputBands * 1.0 / inputBands) * inputBands);
    assert squareOutputBands >= convolutionParams.outputBands : String.format("%d >= %d", squareOutputBands, convolutionParams.outputBands);
    assert squareOutputBands % inputBands == 0 : String.format("%d %% %d", squareOutputBands, inputBands);
    Tensor resultDelta = new Tensor(filterDimensions[0], filterDimensions[1], inputBands * outputBands);
  
    for (int layerNumber = 0; layerNumber < subLayers.size(); layerNumber++) {
      int _layerNumber = layerNumber;
      Tensor deltaTensor = extractor.apply(subLayers.get(layerNumber));
      if (null != deltaTensor) {
        deltaTensor.forEach((v, c) -> {
          int[] coords = c.getCoords();
          int filterBand = getFilterBand(_layerNumber * inputBands * inputBands, coords[2], squareOutputBands);
          if (filterBand < filterDimensions[2]) {
            resultDelta.set(coords[0], coords[1], filterBand, v);
          }
        }, false);
      }
    }
    return resultDelta;
  }
  
  /**
   * Gets filter band.
   *
   * @param filterBandOffset  the filter band offset
   * @param cellFilterBand    the filter band
   * @param squareOutputBands the square output bands
   * @return the filter band
   */
  public int getFilterBand(int filterBandOffset, int cellFilterBand, int squareOutputBands) {
    int inputBands = getInputBands();
    assert cellFilterBand >= 0;
    assert cellFilterBand < (inputBands * inputBands);
    assert filterBandOffset < (inputBands * squareOutputBands);
    int filterBand = cellFilterBand + filterBandOffset;
    filterBand = Coordinate.transposeXY(inputBands, convolutionParams.outputBands, filterBand);
    return filterBand;
  }
  
  /**
   * Gets input bands.
   *
   * @return the input bands
   */
  public int getInputBands() {
    return this.toBand - this.fromBand;
  }
  
  /**
   * Read tensor.
   *
   * @param deltaSet the delta set
   * @param remove   the remove
   * @return the tensor
   */
  public Tensor read(DeltaSet<NNLayer> deltaSet, boolean remove) {
    return read((sublayer) -> {
      final Delta<NNLayer> subnetDelta = remove ? deltaSet.getMap().remove(sublayer) : deltaSet.getMap().get(sublayer);
      if (null == subnetDelta) throw new RuntimeException("No Delta for " + sublayer);
      return new Tensor(subnetDelta.getDelta(), sublayer.kernel.getDimensions());
    });
  }
  
  /**
   * Read tensor.
   *
   * @return the tensor
   */
  public Tensor read() {
    return read((sublayer) -> {
      return sublayer.kernel;
    });
  }
  
  /**
   * Add dag node.
   *
   * @param input the input
   * @return the dag node
   */
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
                         subLayers.stream().map(l -> network.add(l, input)).toArray(i -> new DAGNode[i]));
    }
    if (this.convolutionParams.paddingX != null || this.convolutionParams.paddingY != null) {
      int x = ((filterDimensions[0] - 1) / 2);
      if (this.convolutionParams.paddingX != null) x = this.convolutionParams.paddingX - x;
      int y = ((filterDimensions[1] - 1) / 2);
      if (this.convolutionParams.paddingY != null) y = this.convolutionParams.paddingY - y;
      head = network.add(new ImgZeroPaddingLayer(x, y).setPrecision(convolutionParams.precision), head);
    }
    return head;
  }
  
  @Override
  public String toString() {
    return "ExplodedConvolutionLeg{" +
      "fromBand=" + fromBand +
      ", toBand=" + toBand +
      '}';
  }
}
