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
import com.simiacryptus.mindseye.lang.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * The higher level of convolution construction logic. Provides support for large numbers of input bands by splitting
 * the network into sub-networks that consider only a subset of the input bands, then summing the results together. This
 * strategy remains valid so long as the sub-networks are purely linear.
 */
class ExplodedConvolutionGrid extends ReferenceCountingBase {
  private static final Logger log = LoggerFactory.getLogger(ExplodedConvolutionGrid.class);
  
  /**
   * The Sub layers.
   */
  public final List<ExplodedConvolutionLeg> subLayers = new ArrayList<>();
  /**
   * The Convolution params.
   */
  @javax.annotation.Nonnull
  public final ConvolutionParams convolutionParams;
  
  /**
   * Instantiates a new Exploded network.
   *
   * @param convolutionParams the convolution params
   * @param maxBandBatch      the max band batch
   */
  public ExplodedConvolutionGrid(@javax.annotation.Nonnull ConvolutionParams convolutionParams, int maxBandBatch) {
    this.convolutionParams = convolutionParams;
    for (int fromBand = 0; fromBand < convolutionParams.inputBands; fromBand += maxBandBatch) {
      int toBand = Math.min(convolutionParams.inputBands, fromBand + maxBandBatch);
      subLayers.add(new ExplodedConvolutionLeg(convolutionParams, fromBand, toBand));
    }
  }
  
  @Override
  protected void _free() {
    subLayers.stream().forEach(x -> x.freeRef());
    super._free();
  }
  
  /**
   * Write exploded convolution grid.
   *
   * @param filter the kernel
   * @return the exploded convolution grid
   */
  @javax.annotation.Nonnull
  public ExplodedConvolutionGrid write(@javax.annotation.Nonnull Tensor filter) {
    if (1 == subLayers.size()) {
      subLayers.get(0).write(filter);
    }
    else {
      for (@javax.annotation.Nonnull ExplodedConvolutionLeg leg : subLayers) {
        @javax.annotation.Nonnull int[] legDims = {convolutionParams.masterFilterDimensions[0], convolutionParams.masterFilterDimensions[1], leg.getInputBands() * convolutionParams.outputBands};
        Tensor tensor = new Tensor(legDims).mapCoords(c -> {
          int[] coords = c.getCoords();
          return filter.get(coords[0], coords[1], getFilterBand(leg, coords[2]));
        }, false);
        leg.write(tensor);
        tensor.freeRef();
      }
    }
    return this;
  }
  
  /**
   * Read tensor.
   *
   * @param extractor the extractor
   * @return the tensor
   */
  public Tensor read(@javax.annotation.Nonnull Function<ExplodedConvolutionLeg, Tensor> extractor) {
    if (1 == subLayers.size()) {
      return extractor.apply(subLayers.get(0));
    }
    else {
      @javax.annotation.Nonnull final Tensor filterDelta = new Tensor(convolutionParams.masterFilterDimensions);
      for (@javax.annotation.Nonnull ExplodedConvolutionLeg leg : subLayers) {
        extractor.apply(leg).forEach((v, c) -> {
          int[] coords = c.getCoords();
          filterDelta.set(coords[0], coords[1], getFilterBand(leg, coords[2]), v);
        }, false);
      }
      return filterDelta;
    }
  }
  
  /**
   * Read tensor.
   *
   * @return the tensor
   */
  public Tensor read() {
    return read(l -> l.read());
  }
  
  /**
   * Read tensor.
   *
   * @param deltaSet the delta set
   * @param remove   the remove
   * @return the tensor
   */
  public Tensor read(@javax.annotation.Nonnull DeltaSet<NNLayer> deltaSet, boolean remove) {
    return read(l -> l.read(deltaSet, remove));
  }
  
  private int getFilterBand(@javax.annotation.Nonnull ExplodedConvolutionLeg leg, int legFilterBand) {
    int filterBand = legFilterBand;
    filterBand = filterBand + convolutionParams.outputBands * leg.fromBand;
    return filterBand;
  }
  
  /**
   * Gets network.
   *
   * @return the network
   */
  @javax.annotation.Nonnull
  public PipelineNetwork getNetwork() {
    @javax.annotation.Nonnull PipelineNetwork network = new PipelineNetwork(1);
    add(network.getInput(0));
    return network;
  }
  
  /**
   * Add dag node.
   *
   * @param input the input
   * @return the dag node
   */
  public DAGNode add(@javax.annotation.Nonnull DAGNode input) {
    if (subLayers.size() == 1) {
      return subLayers.get(0).add(input);
    }
    else {
      DAGNetwork network = input.getNetwork();
      List<DAGNode> legs = subLayers.stream().map((ExplodedConvolutionLeg l) -> {
        return l.add(network.wrap(new ImgBandSelectLayer(l.fromBand, l.toBand).setPrecision(convolutionParams.precision), input));
      }).collect(Collectors.toList());
      return network.wrap(new SumInputsLayer().setPrecision(convolutionParams.precision), legs.stream().toArray(i -> new DAGNode[i]));
      //return network.add(new BinarySumLayer().setPrecision(convolutionParams.precision), legs.stream().toArray(i -> new DAGNode[i]));
    }
  }
  
  
}
